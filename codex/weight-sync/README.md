Efficient Weight Sync via CUDA IPC — Implementation Notes

See https://chatgpt.com/share/68bb6fe6-6870-8011-85fe-db83955a57df

Overview
- This folder documents how slime integrates CUDA IPC to update rollout (inference) weights directly from training processes without host copies.
- It summarizes the flow in this repo, cites the referenced blog post, and provides runnable minimal demos of the CUDA IPC serializer path used here.

References
- Blog: Efficient Reinforcement Learning Training — Optimizing Weight Synchronization in slime (hebiao064.github.io/rl-weight-sync)
- Repo docs: docs/en/blogs/release_v0.1.0.md (section “Parameter Update Optimization”).

What to Read
- trace.md — line-by-line annotated code walk-through with pathnames + line numbers.
- demos/minimal_ipc_demo.py — 2‑process demo using SGLang’s MultiprocessingSerializer to share CUDA tensors via IPC.
- demos/minimal_flattened_bucket_demo.py — optional demo using FlattenedTensorBucket if available.
- demos/trace_ipc_handles.py — prints reduce_storage source lines calling storage._share_cuda_ and a live call stack.
 - shims/ — LD_PRELOAD tracer for cudaIpc* calls + helper runner.

Quick Start
- Requirements: Python 3.10+, PyTorch with CUDA, NVIDIA driver, and sglang>=0.3.0 installed in the environment.
- Important: VMM allocators and CUDA IPC are incompatible; ensure LD_PRELOAD‑based allocators are disabled when running IPC demos. For example:
  - Bash: `unset LD_PRELOAD`
  - Inline: `env -u LD_PRELOAD python codex/weight-sync/demos/minimal_ipc_demo.py`

Run
1) Basic IPC demo
   python codex/weight-sync/demos/minimal_ipc_demo.py --num-tensors 3 --numel 1048576

2) Flattened bucket demo (if sglang.srt.model_executor FlattenedTensorBucket is available)
   python codex/weight-sync/demos/minimal_flattened_bucket_demo.py

3) Trace IPC handle creation (requires CUDA + sglang)
   unset LD_PRELOAD
   python codex/weight-sync/demos/trace_ipc_handles.py --numel 1048576 --num-tensors 2

4) OS-level (LD_PRELOAD) trace of cudaIpc* calls
   # build the shim
   make -C codex/weight-sync/shims

   # run any of the demos with tracing enabled
   codex/weight-sync/shims/run_with_ipc_trace.sh \
       python codex/weight-sync/demos/minimal_ipc_demo.py --num-tensors 2 --numel 65536

   # check log (default /tmp/ipc.log)
   python codex/weight-sync/shims/check_ipc_log.py /tmp/ipc.log

Both demos print checksums from the receiver, then the sender mutates the original CUDA tensors and the receiver observes the change without re‑sending data — demonstrating zero‑copy CUDA IPC mapping.

Notes
- These demos do not start the full HTTP server in sglang. They exercise the exact serializer/deserializer used by this repo during weight updates.
- If you don’t have GPUs, the demos will fall back to CPU and still run, but that path does NOT use CUDA IPC.
