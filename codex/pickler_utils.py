# mp_pickler_inspect.py
import os
import types
from multiprocessing.reduction import ForkingPickler

def _qname(obj):
    if isinstance(obj, types.FunctionType):
        return f"{obj.__module__}.{obj.__name__} (id={id(obj)})"
    mod = getattr(obj, "__module__", "<no_mod>")
    name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    return f"{mod}.{name}"

def dump_dispatch_table(filter_prefixes=None):
    """
    Print the ForkingPickler.dispatch table.
    Set filter_prefixes=['torch'] to only show torch-related entries.
    """
    table = getattr(ForkingPickler, "dispatch", {})
    rows = []
    for typ, fn in table.items():
        tname = _qname(typ)
        if filter_prefixes and not any(tname.startswith(p) for p in filter_prefixes):
            continue
        rows.append((tname, _qname(fn)))
    rows.sort(key=lambda x: x[0].lower())
    width = max((len(r[0]) for r in rows), default=0)
    print(f"[pid={os.getpid()}] ForkingPickler.dispatch entries ({len(rows)} shown):")
    for tname, fname in rows:
        print(f"  {tname.ljust(width)}  ->  {fname}")

def show_tensor_reducer_summary():
    """Show the current reducer functions for Tensor/Storage types."""
    import torch
    import torch.multiprocessing.reductions as red

    cur_tensor = ForkingPickler.dispatch.get(torch.Tensor)
    print(f"[pid={os.getpid()}] torch.Tensor reducer = {_qname(cur_tensor)}")

    # Storage types differ across versions; cover both.
    stor_types = []
    try:
        # Newer
        from torch import _UntypedStorage as UntypedStorage
        stor_types.append(UntypedStorage)
    except Exception:
        pass
    for cand in ("TypedStorage", "_TypedStorage"):
        try:
            stor_types.append(getattr(torch.storage, cand))
        except Exception:
            pass

    for st in stor_types:
        cur = ForkingPickler.dispatch.get(st)
        print(f"[pid={os.getpid()}] {st.__module__}.{st.__name__} reducer = {_qname(cur)}")

def install_logging_reducers():
    """
    Patch tensor/storage reducers with logging wrappers AND re-register them
    so ForkingPickler uses the wrappers (critical).
    Call this in every process BEFORE any pickling happens.
    """
    import torch
    import torch.multiprocessing.reductions as red

    # Keep originals
    _orig_reduce_tensor = red.reduce_tensor
    _orig_reduce_storage = red.reduce_storage

    def reduce_tensor_logged(t):
        print(f"[pid={os.getpid()}][reduce_tensor] tensor {tuple(t.size())} {t.dtype} dev={t.device}")
        return _orig_reduce_tensor(t)

    def reduce_storage_logged(storage):
        # Works for both typed/untyped storage objects
        dev = getattr(storage, "device", None)
        print(f"[pid={os.getpid()}][reduce_storage] {type(storage).__name__} dev={dev} size={len(storage)}")
        return _orig_reduce_storage(storage)

    # Monkey-patch module attributes (nice-to-have)
    red.reduce_tensor = reduce_tensor_logged
    red.reduce_storage = reduce_storage_logged

    # **Re-register** so ForkingPickler actually uses these wrappers.
    ForkingPickler.register(torch.Tensor, red.reduce_tensor)

    # Storage classes vary by PyTorch version. Register all that exist.
    try:
        from torch import _UntypedStorage as UntypedStorage
        ForkingPickler.register(UntypedStorage, red.reduce_storage)
    except Exception:
        pass
    for name in ("TypedStorage", "_TypedStorage"):
        try:
            ST = getattr(torch.storage, name)
            ForkingPickler.register(ST, red.reduce_storage)
        except Exception:
            pass

    # Show what we ended up wiring
    print(f"[pid={os.getpid()}] After (re)register:")
    show_tensor_reducer_summary()
