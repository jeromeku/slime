CUDA_VISIBLE_DEVICES=0,1 python ipc_demo.py

# Install Nsight Systems and run:
# nsys profile -t cuda,osrt -f true -o ipc_trace --capture-range=nvtx --capture-range-end=stop \
#     python ipc_demo.py
# # or simpler, without NVTX capture range:
# nsys profile -t cuda,osrt -f true -o ipc_trace python ipc_demo.py
# Then open ipc_trace.qdrep and filter for the cudaIpc* calls
