#!/usr/bin/env python3
"""Simple parser to assert that cudaIpcGetMemHandle/Open were called.

Usage:
  python codex/weight-sync/shims/check_ipc_log.py /tmp/ipc.log
"""
from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        print("usage: check_ipc_log.py /path/to/ipc.log")
        sys.exit(2)
    p = sys.argv[1]
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    got_get = "cudaIpcGetMemHandle(" in s
    got_open = "cudaIpcOpenMemHandle(" in s
    print("cudaIpcGetMemHandle:", got_get)
    print("cudaIpcOpenMemHandle:", got_open)
    if not (got_get and got_open):
        print("ERROR: expected IPC calls were not observed.")
        sys.exit(1)
    print("OK: observed IPC calls.")


if __name__ == "__main__":
    main()

