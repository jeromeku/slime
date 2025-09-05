#!/usr/bin/env python3
"""
Optional: FlattenedTensorBucket demo

Shows how a list of (name, tensor) pairs can be flattened into a single big CUDA tensor plus metadata,
serialized via MultiprocessingSerializer, and reconstructed in another process.

This mirrors the fast path used when `FlattenedTensorBucket` is available.
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import torch


def sender(q: mp.Queue):
    try:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
        from sglang.srt.utils import MultiprocessingSerializer
    except Exception as e:
        print("FlattenedTensorBucket not available:", e)
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    named = [("w0", torch.randn(16, 16, device=device)), ("w1", torch.randn(8, 8, device=device))]
    bucket = FlattenedTensorBucket(named_tensors=named)
    meta = bucket.get_metadata()
    flat = bucket.get_flattened_tensor()
    payload = MultiprocessingSerializer.serialize({"flattened_tensor": flat, "metadata": meta}, output_str=True)
    q.put(payload)


def receiver(q: mp.Queue):
    from sglang.srt.utils import MultiprocessingSerializer
    import json

    s = q.get(timeout=30)
    restored = MultiprocessingSerializer.deserialize(s)
    # Expect a dict with keys flattened_tensor + metadata, just print simple info
    if isinstance(restored, dict):
        flat = restored["flattened_tensor"]
        meta = restored["metadata"]
        print("Restored flat device:", flat.device)
        print("Num elements:", flat.numel())
        print("First entry in metadata:", json.dumps(meta[0]) if meta else None)
    else:
        print("Unexpected deserialized type:", type(restored))


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    r = ctx.Process(target=receiver, args=(q,))
    r.start()
    try:
        sender(q)
    finally:
        r.join(timeout=30)


if __name__ == "__main__":
    main()

