#!/usr/bin/env python3
"""
Minimal CUDA IPC demo mirroring slime's serializer path.

What it does
- Sender creates a few CUDA tensors, serializes them via sglang.srt.utils.MultiprocessingSerializer (IPC handles),
  and passes the serialized strings to a separate Python process.
- Receiver deserializes into CUDA tensors that map the sender's storage via CUDA IPC, prints checksums.
- Sender then mutates the original tensors in-place; receiver re-checks without any re-send to show zero-copy mapping.

Requirements
- torch with CUDA, sglang>=0.3.0
- Ensure LD_PRELOAD virtual memory allocators are DISABLED when testing CUDA IPC.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import List, Tuple

try:
    import torch
except Exception as e:  # pragma: no cover
    print("PyTorch import failed:", e)
    sys.exit(1)


def require_cuda_ipc() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU (no IPC).")
        return
    if os.environ.get("LD_PRELOAD"):
        print("Warning: LD_PRELOAD is set; CUDA IPC may be incompatible with VMM allocators.")


def sender(num_tensors: int, numel: int, q: mp.Queue, ready: mp.Event, again: mp.Event):
    from sglang.srt.utils import MultiprocessingSerializer  # import in-process

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors: List[Tuple[str, torch.Tensor]] = []
    for i in range(num_tensors):
        t = torch.arange(numel, device=device, dtype=torch.float32) + i
        tensors.append((f"w{i}", t))

    # Serialize to strings (this is what slime sends over the wire)
    payload = MultiprocessingSerializer.serialize(tensors, output_str=True)
    q.put(payload)

    # Wait until receiver reports first checksum
    ready.wait()

    # Mutate in-place on the sender; receiver should observe changes via the IPC mapping
    if device == "cuda":
        for name, t in tensors:
            t.add_(1.0)
        # give receiver time to read again
        time.sleep(0.2)
        again.set()
    else:
        # CPU path uses pickled bytes, so this mutation does not reflect on receiver
        again.set()


def receiver(q: mp.Queue, ready: mp.Event, again: mp.Event):
    from sglang.srt.utils import MultiprocessingSerializer

    serialized = q.get(timeout=30)
    objs = MultiprocessingSerializer.deserialize(serialized)
    # Expect list[tuple[str, Tensor]]
    assert isinstance(objs, list) and isinstance(objs[0], tuple)
    names = [n for n, _ in objs]
    checksums1 = {n: v.sum().item() for n, v in objs}
    print("Receiver checksums (first):", checksums1)
    ready.set()

    # Wait for sender's in-place mutation
    again.wait(timeout=30)
    checksums2 = {n: v.sum().item() for n, v in objs}
    print("Receiver checksums (second):", checksums2)

    # On CUDA, sums should increase by numel for every tensor (since we add 1.0 per element)
    if objs[0][1].is_cuda:
        diffs = {n: checksums2[n] - checksums1[n] for n in names}
        print("Checksum deltas:", diffs)
        assert all(abs(d - objs[0][1].numel()) < 1e-3 for d in diffs.values()), (
            "IPC mapping not reflecting mutations; check LD_PRELOAD/driver."
        )
    else:
        print("CPU fallback path: mutation is not reflected (no IPC).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tensors", type=int, default=2)
    parser.add_argument("--numel", type=int, default=1 << 20)
    args = parser.parse_args()

    require_cuda_ipc()

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ready = ctx.Event()
    again = ctx.Event()
    r = ctx.Process(target=receiver, args=(q, ready, again))
    r.start()
    try:
        sender(args.num_tensors, args.numel, q, ready, again)
    finally:
        r.join(timeout=30)
        if r.exitcode not in (0, None):
            sys.exit(r.exitcode)


if __name__ == "__main__":
    main()

