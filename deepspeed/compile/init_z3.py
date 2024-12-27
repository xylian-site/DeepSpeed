# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses

from .passes import zero3_compile, prefetch, selective_gather
from .backend import make_backend, launch_compile_passes, init_schedule, opt_passes
from .patch_fake_tensor import patch_fake_tensor
from .util import log_rank0

WARMUP = 5


def init_z3(engine, compile_config, compile_kwargs, schedule=None):

    if engine.optimizer is not None and hasattr(engine.optimizer,
                                                '_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer'):
        engine.optimizer._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer = None
        get_accelerator().empty_cache()
    engine.nz3.init(engine.data_parallel_group, engine.zero_reduce_bucket_size(), compile_config.double_buffer,
                    compile_config.symmetric_memory)

    # Unset hooks
    for m in engine.module.modules():
        m._parameters = m._original_parameters
    engine.optimizer.parameter_offload._remove_module_hooks()

    for hook in engine.optimizer._grad_acc_hooks:
        hook.remove()
    engine.optimizer._grad_acc_hooks.clear()

    # Unpatch linear
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk

    if compile_config.symmetric_memory:
        group_name = engine.data_parallel_group.group_name
        dist.enable_symm_mem_for_group(group_name)

    for p in engine.module.parameters():
        grad_buffer = engine.optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        engine.nz3.register_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist)

    if schedule is None:
        schedule = []
        schedule.append((0, [zero3_compile.add_z3_gather_release]))
        schedule.append(
            (WARMUP,
             [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather]))
    else:

        def passes_name_to_fn(passes):
            for pass_name in passes:
                assert pass_name in opt_passes, f"Unknown pass {pass_name}"
            return [opt_passes[pass_name] for pass_name in passes]

        schedule = [(step, passes_name_to_fn(passes)) for step, passes in schedule]

    init_schedule(schedule)

    log_rank0(f"Opt passes: {schedule}")
    engine.launch_compile_passes = launch_compile_passes

    patch_fake_tensor()
    return make_backend(compile_kwargs=compile_kwargs, free_activation=compile_config.free_activation)
