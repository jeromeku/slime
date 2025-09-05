#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LIB="$THIS_DIR/libcuda_ipc_trace.so"

if [[ ! -f "$LIB" ]]; then
  echo "Build first: make -C $THIS_DIR" >&2
  exit 1
fi

export LD_PRELOAD="$LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export CUDA_IPC_TRACE_LOG="./ipc_trace.log"
export CUDA_IPC_TRACE_STACK=1
exec "$@"

