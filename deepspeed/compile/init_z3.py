# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed import comm as dist
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses

from .passes.prefetch import schedule_prefetch
from .passes.selective_gather import make_selective_gather
from .passes.offload_adam_states import init_offload_opt_states, move_offload_opt_states
from .stage3_backend import make_stage3_backend, launch_opt_passes
from .patch_compiled_func import patch_compiled_func
from .patch_fake_tensor import patch_fake_tensor


def init_z3(engine, compile_config, compile_kwargs, passes=None):

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

    WARMUP_STEPS = 5

    if passes is None:
        passes = ["prefetch", "selective_gather"]

    opt_passes = []
    if "prefetch" in passes:
        opt_passes.append((schedule_prefetch, 0.0))
    if "selective_gather" in passes:
        opt_passes.append((make_selective_gather(engine.optimizer, engine.nz3), -1.0))

    if compile_config.offload_opt_states:
        init_offload_opt_states(engine.optimizer.optimizer, engine.nz3)
        opt_passes = [(move_offload_opt_states, 0.7)]

    if engine.global_rank == 0:
        print(f"Opt passes: {opt_passes}")

    def launch_compile_passes(micro_steps=engine.micro_steps,
                              global_steps=engine.global_steps,
                              update=engine.is_gradient_accumulation_boundary()):
        if global_steps == WARMUP_STEPS and engine.micro_steps % engine.gradient_accumulation_steps() == 0:
            torch._dynamo.reset()
            engine.nz3.reset()
            patch_compiled_func()
            launch_opt_passes()

    engine.launch_compile_passes = launch_compile_passes

    patch_fake_tensor()
    return make_stage3_backend(opt_passes,
                               compile_kwargs=compile_kwargs,
                               free_activation=compile_config.free_activation,
                               offload_activation=compile_config.offload_activation,
                               offload_opt_states=compile_config.offload_opt_states,
                               dump_graphs=compile_config.dump_graphs)
