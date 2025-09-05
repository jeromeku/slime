Line‑By‑Line Trace: CUDA IPC Weight Sync Integration

Context
- Training processes produce tensors on GPU; rollout engines (inference) live in separate processes.
- Instead of copying weights over CPU/PCIe, slime serializes CUDA tensor storages into IPC handles and sends only lightweight metadata; rollout engines open those handles and copy device‑to‑device directly on GPU.

Key Files

1) slime/backends/fsdp_utils/update_weight_utils.py

   22–38 connect_rollout_engines: define per‑engine gather group and record the target rollout engine for IPC handoff
   ```py
   # slime/backends/fsdp_utils/update_weight_utils.py:22-38
   def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
       self.rollout_engines = rollout_engines

       # Here we assume the gpu id of rollout engines and train actors are the same.
       for i, engine in enumerate(self.rollout_engines):
           start_rank = i * self.args.rollout_num_gpus_per_engine
           end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
           group_ranks = list(range(start_rank, end_rank))
           new_group = dist.new_group(
               ranks=group_ranks,
               backend="gloo",
           )
           if dist.get_rank() in group_ranks:
               self._ipc_gather_src = start_rank
               self._ipc_gather_group = new_group
               self._ipc_engine = engine
   ```

   39–56 update_weights: materialize full state dict and serialize CUDA tensors with sglang’s MultiprocessingSerializer
   ```py
   # slime/backends/fsdp_utils/update_weight_utils.py:39-56
   @torch.no_grad()
   def update_weights(self):
       monkey_patch_torch_reductions()
       with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
           named_tensors = [(name, param) for name, param in self.model.state_dict().items()]

       if use_flattened_tensor_bucket:
           flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
           metadata = flattened_tensor_bucket.get_metadata()

           flattened_tensor_data = {
               "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
               "metadata": metadata,
           }
           serialized_tensors = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
       else:
           serialized_tensors = MultiprocessingSerializer.serialize(named_tensors, output_str=True)
   ```

   57–75 gather and RPC into rollout engine: rank‑local gather to `_ipc_gather_src`, then invoke `engine.update_weights_from_tensor`
   ```py
   # slime/backends/fsdp_utils/update_weight_utils.py:57-75
   serialized_named_tensors = (
       [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
   )
   dist.gather_object(
       serialized_tensors,
       object_gather_list=serialized_named_tensors,
       dst=self._ipc_gather_src,
       group=self._ipc_gather_group,
   )

   if dist.get_rank() == self._ipc_gather_src:
       kwargs = {"serialized_named_tensors": serialized_named_tensors}
       if use_flattened_tensor_bucket:
           kwargs["load_format"] = "flattened_bucket"

       ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
       ray.get(ref)
   ```

2) slime/backends/megatron_utils/update_weight_utils.py

   307–323 Megatron path: mirror of FSDP connect; sets `_ipc_engine`, `_ipc_gather_src`, `_ipc_gather_group`
   ```py
   # slime/backends/megatron_utils/update_weight_utils.py:307-323
   def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
       self.rollout_engines = rollout_engines
       # assume the gpu id of rollout engines and train actors are the same
       for i, engine in enumerate(self.rollout_engines):
           start_rank = i * self.args.rollout_num_gpus_per_engine
           end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
           group_ranks = list(range(start_rank, end_rank))
           new_group = dist.new_group(ranks=group_ranks, backend="gloo")
           if dist.get_rank() in group_ranks:
               self._ipc_gather_src = start_rank
               self._ipc_gather_group = new_group
               self._ipc_engine = engine
   ```

   335–401 Compute tensors to sync (gather/convert); then batch into serializer
   ```py
   # slime/backends/megatron_utils/update_weight_utils.py:335-401
   def _update_bucket_weights_from_tensor(self, param_infos):
       monkey_patch_torch_reductions()
       # Build tensors on the correct CUDA device
       params = []
       for info in param_infos:
           if dist.get_rank() == info.src_rank:
               params.append(torch.nn.Parameter(
                   self.weights["actor"][info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                   requires_grad=False,
               ))
           else:
               params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
       torch.cuda.synchronize()
       # Broadcast and all‑gather omitted here for brevity
       converted_named_tensors = []
       for info, param in zip(param_infos, gathered_params):
           param = remove_padding(info.name, param, self.vocab_size)
           converted_named_tensors.extend(convert_to_hf(...))
       self._update_converted_params_from_tensor(converted_named_tensors)
   ```

   402–433 Serialize and hand off to rollout engine
   ```py
   # slime/backends/megatron_utils/update_weight_utils.py:402-433
   if use_flattened_tensor_bucket and self.quantization_config is None:
       flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=converted_named_tensors)
       metadata = flattened_tensor_bucket.get_metadata()
       flattened_tensor_data = {
           "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
           "metadata": metadata,
       }
       serialized_tensors = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
   else:
       serialized_tensors = MultiprocessingSerializer.serialize(converted_named_tensors, output_str=True)

   serialized_named_tensors = (
       [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
   )
   dist.gather_object(...)
   if dist.get_rank() == self._ipc_gather_src:
       kwargs = {"serialized_named_tensors": serialized_named_tensors}
       if use_flattened_tensor_bucket and self.quantization_config is None:
           kwargs["load_format"] = "flattened_bucket"
       ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
       ray.get(ref)
   ```

3) slime/backends/sglang_utils/sglang_engine.py

   149–169 rollout engine entrypoint: bridges to sglang HTTP server, which performs CUDA‑to‑CUDA copy using IPC‑opened memory
   ```py
   # slime/backends/sglang_utils/sglang_engine.py:149-169
   def update_weights_from_tensor(self, serialized_named_tensors: List[str], load_format: Optional[str] = None, flush_cache: bool = False):
       """
       Update model weights from tensor data. The HTTP server will only post meta data,
       and the real weights will be copied directly from GPUs.
       Note: The model should be on GPUs rather than CPU for this functionality to work properly.
       """
       return self._make_request(
           "update_weights_from_tensor",
           {"serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache},
       )
   ```

How CUDA IPC is used
- MultiprocessingSerializer.serialize turns CUDA tensors into lightweight strings carrying IPC handles and metadata (device, dtype, shape, strides, storage offsets). The tensors’ storage is not copied; instead, cudaIpcGetMemHandle references are exported.
- On the rollout side, the HTTP server (in sglang) deserializes those strings, opens the CUDA IPC handles (cudaIpcOpenMemHandle), builds torch tensors pointing to the mapped storage, and performs an in‑GPU copy into the serving model’s parameter buffers.
- Important: slime’s docs note that VMM allocators are incompatible with CUDA IPC; disable LD_PRELOAD replacement when doing weight sync and DeepEP.

Cross‑refs
- docs/en/blogs/release_v0.1.0.md:82 — VMM APIs and cudaIPC APIs (e.g., cudaIpcGetMemHandle) are incompatible; switch back to cudaMalloc for IPC‑based sync.
- Blog: Efficient Reinforcement Learning Training — Optimizing Weight Synchronization in slime — architectural background and performance results.

