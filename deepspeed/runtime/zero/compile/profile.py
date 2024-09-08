# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from typing import Any
import statistics

import torch
from torch.utils._pytree import tree_map
from torch.fx import GraphModule, Interpreter

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .util import is_comm_op


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, iteration: int = 10, warmup: int = 5):
        super().__init__(gm)

        assert iteration > 0
        assert warmup >= 0
        assert warmup < iteration
        self.iteration = iteration
        self.warmup = warmup

    def run_node(self, n: torch.fx.Node) -> Any:
        fake_ret = super().run_node(n)

        if n.op in {"placeholder", "output"}:
            n.meta["device_time"] = 0.0
            n.meta["wall_time"] = 0.0
        else:
            accelerator = get_accelerator()
            start_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]
            end_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]

            device = torch.device(accelerator.current_device())

            def to_device(t):
                if isinstance(t, torch.Tensor):
                    return t.to(device)
                return t

            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            args = tree_map(to_device, args)
            kwargs = tree_map(to_device, kwargs)
            walltimes = []

            if is_comm_op(n):
                dist.barrier()

            for i in range(self.iteration):
                start = time.time()
                start_events[i].record()
                getattr(self, n.op)(n.target, args, kwargs)
                end_events[i].record()
                walltimes.append(time.time() - start)

            if is_comm_op(n):
                dist.barrier()

            accelerator.synchronize()
            n.meta["device_time"] = statistics.mean([s.elapsed_time(e)
                                                     for s, e in zip(start_events, end_events)][self.warmup:])
            n.meta["wall_time"] = statistics.mean(walltimes[self.warmup:]) * 1000

        return fake_ret
