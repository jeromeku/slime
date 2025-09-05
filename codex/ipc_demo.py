#!/usr/bin/env python3
import io, os, socket, time, pickle, base64 as pybase64
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing.reduction import ForkingPickler

# --- Monkeypatch PyTorch reducers for logging ---
import torch.multiprocessing.reductions as red
from pickler_utils import dump_dispatch_table

# --- Serializer (same API as yours) ---
class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        out = buf.read()
        return pybase64.b64encode(out).decode("utf-8") if output_str else out

    @staticmethod
    def deserialize(data):
        if isinstance(data, str):
            data = pybase64.b64decode(data, validate=True)
        return ForkingPickler.loads(data)


_orig_reduce_tensor = red.reduce_tensor
_orig_rebuild_cuda_tensor = red.rebuild_cuda_tensor

def _short_handle(h):
    try:
        b = bytes(h)
        return f"{len(b)}B:{b[:8].hex()}..."
    except Exception:
        return str(h)

def reduce_tensor_logged(t):
    storage = t._typed_storage()
    devtype = storage._untyped_storage.device.type
    print(f"[{os.getpid()}][reduce_tensor] tensor {tuple(t.size())} {t.dtype} dev={t.device} storage_devtype={devtype}")
    if devtype == "cuda":
        print(f"[{os.getpid()}][reduce_tensor]  -> storage._share_cuda_() (creating cudaIpcMemHandle + refcounter + event)")
    rv = _orig_reduce_tensor(t)
    if devtype == "cuda":
        func, args = rv
        (_, _, _, _, storage_cls, dtype, storage_device, storage_handle,
         storage_size_bytes, storage_offset_bytes, requires_grad,
         ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = args
        print(f"[{os.getpid()}][reduce_tensor]  -> handle={_short_handle(storage_handle)} "
              f"size={storage_size_bytes}B off={storage_offset_bytes} "
              f"ref={_short_handle(ref_counter_handle)}:{ref_counter_offset} "
              f"evt={_short_handle(event_handle)} sync={event_sync_required}")
    return rv

def rebuild_cuda_tensor_logged(*args):
    (tensor_cls, tensor_size, tensor_stride, tensor_offset, storage_cls, dtype,
     storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
     requires_grad, ref_counter_handle, ref_counter_offset, event_handle,
     event_sync_required) = args
    print(f"[{os.getpid()}][rebuild_cuda_tensor] OPEN handle={_short_handle(storage_handle)} "
          f"dev={storage_device} size={storage_size_bytes}B off={storage_offset_bytes} "
          f"ref={_short_handle(ref_counter_handle)}:{ref_counter_offset} evt={_short_handle(event_handle)}")
    return _orig_rebuild_cuda_tensor(*args)

def patch_reductions():
    red.reduce_tensor = reduce_tensor_logged
    red.rebuild_cuda_tensor = rebuild_cuda_tensor_logged
    red.init_reductions()

# --- Demo payload ---
def make_payload(dev):
    # nontrivial view to show size/offset bookkeeping
    a = torch.arange(16, dtype=torch.float32, device=dev).reshape(4,4)[::2, ::2]
    b = torch.randn(3, 5, device=dev)
    return {"name": "demo", "a": a, "b": b, "meta": {"when": time.time(), "host": socket.gethostname()}}

def init_pg(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29876")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def worker(rank, world_size, dst=0):

    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    init_pg(rank, world_size)

    if rank == 0:
        print(f"Before patch:"); dump_dispatch_table(['torch'])
    patch_reductions()
    if rank == 0:
        print(f"After patch:"); dump_dispatch_table(['torch'])
    
    # NVTX range for profiling in Nsight Systems
    try:
        torch.cuda.nvtx.range_push(f"rank{rank}:serialize")
    except Exception:
        pass

    obj = make_payload(dev)
    blob_b64 = MultiprocessingSerializer.serialize(obj, output_str=True)

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass

    if rank == dst:
        gather_list = [None for _ in range(world_size)]
        dist.gather_object(blob_b64, object_gather_list=gather_list, dst=dst)  # dst also sends its own blob

        # IMPORTANT: don't open your own handle; only open remote blobs
        remote_objs = []
        try:
            torch.cuda.nvtx.range_push(f"rank{rank}:deserialize_remote")
        except Exception:
            pass

        for r, blob in enumerate(gather_list):
            if r == dst:
                print(f"[{os.getpid()}][rank{rank}] skipping self-deserialize to avoid opening own IPC handle")
                continue
            # This triggers storage_cls._new_shared_cuda(...) -> cudaIpcOpenMemHandle
            remote = MultiprocessingSerializer.deserialize(blob)
            remote_objs.append(remote)
            print(f"[{os.getpid()}][rank{rank}] opened remote rank{r} tensors; a.device={remote['a'].device}")

        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass

        # Drop references and collect
        del remote_objs
        print(f"[{os.getpid()}][rank{rank}] ipc_collect() complete; hitting barrier")
    else:
        dist.gather_object(blob_b64, object_gather_list=None, dst=dst)
        print(f"[{os.getpid()}][rank{rank}] sent blob to dst; waiting at barrier")
        
    dist.barrier()
    torch.cuda.ipc_collect()
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    world = 2
    procs = []
    for r in range(world):
        p = mp.get_context("spawn").Process(target=worker, args=(r, world))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
