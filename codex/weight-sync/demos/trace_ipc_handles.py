#!/usr/bin/env python3
"""
Trace that MultiprocessingSerializer.serialize triggers CUDA IPC handle creation.

What this prints:
- The exact Python source location of torch.multiprocessing.reductions.reduce_storage and
  the line(s) that call storage._share_cuda_.
- A live call stack when reduce_storage is invoked via MultiprocessingSerializer.serialize.
- The rebuild function chosen (should be rebuild_storage_cuda for CUDA storages), and arg types
  including the opaque CUDA IPC handle object type coming from C++.

Usage:
  unset LD_PRELOAD  # IMPORTANT: VMM/alloc hooks conflict with CUDA IPC
  python codex/weight-sync/demos/trace_ipc_handles.py --numel 1048576 --num-tensors 2

Requires:
  - CUDA-capable PyTorch and GPU present.
  - sglang installed (for sglang.srt.utils.MultiprocessingSerializer).
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback

import torch


def print_reduce_functions_source(verbose=False):
    import torch.multiprocessing.reductions as red
    print("[INFO] reductions.py path:", red.__file__)
    # Show reduce_tensor (CUDA path)
    fn = red.reduce_tensor
    try:
        src_lines, start = inspect.getsourcelines(fn)
        path = inspect.getsourcefile(fn) or inspect.getfile(fn)
        if verbose:
            print("[INFO] torch.multiprocessing.reductions.reduce_tensor source:")
            for i, line in enumerate(src_lines, start):
                print(f"{path}:{i}: {line.rstrip()}")
        idxs = [i for i, l in enumerate(src_lines, start) if "_share_cuda_" in l]
        if idxs:
            print("[INFO] Lines calling storage._share_cuda_:", idxs)
        else:
            print("[WARN] Could not find '_share_cuda_' in reduce_tensor source; PyTorch may have changed.")
    except (OSError, TypeError):
        print("[WARN] Could not retrieve Python source for reduce_tensor (possibly built-in).")
    # Show reduce_storage (CPU path for completeness)
    fn2 = red.reduce_storage
    try:
        src_lines, start = inspect.getsourcelines(fn2)
        path = inspect.getsourcefile(fn2) or inspect.getfile(fn2)
        if verbose:
            print("[INFO] torch.multiprocessing.reductions.reduce_storage source:")
            for i, line in enumerate(src_lines, start):
                print(f"{path}:{i}: {line.rstrip()}")
    except (OSError, TypeError):
        pass


def install_reduction_traces():
    import torch.multiprocessing.reductions as red
    traces = {}

    def wrap(fn_name):
        fn = getattr(red, fn_name)
        def w(*a, **k):
            print(f"[TRACE] {fn_name} called; args types={[type(x) for x in a]}")
            print("[TRACE] Python call stack (most recent last):")
            print("".join(traceback.format_stack(limit=32)))
            out = fn(*a, **k)
            try:
                rebuild_fn, args = out
                rebuild_name = getattr(rebuild_fn, "__name__", str(rebuild_fn))
                print(f"[TRACE] {fn_name} -> {rebuild_name}; arg types={[type(x) for x in args]}")
            except Exception:
                pass
            return out
        setattr(red, fn_name, w)
        traces[fn_name] = fn

    # CUDA tensors go through reduce_tensor; CPU storages through reduce_storage
    wrap("reduce_tensor")
    wrap("reduce_storage")
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numel", type=int, default=1 << 20)
    ap.add_argument("--num-tensors", type=int, default=2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. This trace requires a CUDA GPU.")
        sys.exit(1)
    if os.environ.get("LD_PRELOAD"):
        print("[WARN] LD_PRELOAD is set; VMM hooks may break CUDA IPC. Consider 'unset LD_PRELOAD'.")

    try:
        from sglang.srt.utils import MultiprocessingSerializer
    except Exception as e:
        print("[ERROR] Could not import sglang.srt.utils.MultiprocessingSerializer:", e)
        print("Install sglang and try again.")
        sys.exit(1)

    print_reduce_functions_source()
    install_reduction_traces()

    # Create CUDA tensors and serialize using the same path as in slime
    named_tensors = [(f"w{i}", (torch.arange(args.numel, device="cuda", dtype=torch.float32) + i)) for i in range(args.num_tensors)] 
    payload = MultiprocessingSerializer.serialize(
        named_tensors,
        output_str=True,
    )
    print("[OK] Serialization complete; payload length:", len(payload))
    names, tensors = zip(*named_tensors)
    for t in tensors:
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    main()
