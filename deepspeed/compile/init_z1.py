# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .passes import zero1_compile
from .backend import make_backend, launch_compile_passes, init_schedule
from .util import log_rank0

WARMUP = 5


def init_z1(engine, compile_config, compile_kwargs, schedule=None):

    for hook in engine.optimizer._grad_acc_hooks:
        hook.remove()
    engine.optimizer._grad_acc_hooks.clear()

    for i, p in enumerate(engine.module.parameters()):
        p.param_id = i

    if schedule is None:
        schedule = []
        schedule.append((0, [zero1_compile.add_z1_reduce]))

    init_schedule(schedule)

    log_rank0(f"Opt passes: {schedule}")
    engine.launch_compile_passes = launch_compile_passes
    return make_backend(compile_kwargs=compile_kwargs, free_activation=False)
