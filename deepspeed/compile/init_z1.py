# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.accelerator import get_accelerator

from .passes import zero1_compile
from .backend import make_backend, launch_compile_passes, init_schedule
from .util import log_rank0, get_deepcompile_handle

WARMUP = 5


def init_z1(engine, compile_config, compile_kwargs, schedule=None):

    optimizer = engine.optimizer
    for hook in optimizer._grad_acc_hooks:
        hook.remove()
    optimizer._grad_acc_hooks.clear()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, engine.zero_reduce_bucket_size(), compile_config.double_buffer,
            compile_config.symmetric_memory)

    grad_buffer = {}

    for i, group in enumerate(optimizer.bit16_groups):

        grad_buffer[i] = optimizer.get_flat_partition(optimizer.params_in_partition[i],
                                                      optimizer.first_offset[i],
                                                      optimizer.partition_size[i],
                                                      dtype=optimizer.gradient_accumulation_dtype,
                                                      device=get_accelerator().current_device_name(),
                                                      return_tensor_list=True)
        grad_buffer[i] = [p.clone().detach() for p in grad_buffer[i]]  # Maybe not necessary

        index_in_partition = 0
        first_in_partition = True
        for p in group:
            param_id = optimizer.get_param_id(p)
            p.param_id = param_id
            in_partition = optimizer.is_param_in_current_partition[param_id]

            if in_partition:
                buf = grad_buffer[i][index_in_partition]
                offset = optimizer.first_offset[i] if first_in_partition else 0
                # print(f"[r{dist.get_rank()}] Registering group {i} param {param_id} in_partition={in_partition} p={p.shape} buf={buf.shape} partition_offset={offset}")
                dc.register_z1_param(p.param_id, p.shape, p, buf, int(offset))
                index_in_partition += 1
                first_in_partition = False
            else:
                # print(f"[r{dist.get_rank()}] Registering group {i} param {param_id} in_partition={in_partition} p={p.shape} buf=None")
                dc.register_z1_param(p.param_id, p.shape, p, torch.empty([0], dtype=p.dtype, device=p.device), 0)

    if schedule is None:
        schedule = []
        schedule.append((0, [zero1_compile.add_z1_reduce]))

    init_schedule(schedule)

    log_rank0(f"Opt passes: {schedule}")
    engine.launch_compile_passes = launch_compile_passes
    return make_backend(compile_kwargs=compile_kwargs, free_activation=False)
