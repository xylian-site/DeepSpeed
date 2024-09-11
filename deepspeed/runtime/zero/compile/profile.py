# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from typing import Any
import statistics

import torch
from torch.utils._pytree import tree_map, tree_all
from torch.fx import GraphModule, Interpreter
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from .util import is_comm_op


def _can_be_materialized(v):
    return isinstance(v, torch.Tensor) and v.is_floating_point()


def _materialize(t, device):
    if _can_be_materialized(t):
        return torch.randn(t.shape, dtype=t.dtype, layout=t.layout, device=device)
    return t


def _can_all_args_be_materialized(v):
    return tree_all(lambda x: not torch.is_tensor(x) or _can_be_materialized(x), v)


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

            def materialize(t):
                return _materialize(t, device)

            with maybe_disable_fake_tensor_mode():
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert isinstance(args, tuple)
                assert isinstance(kwargs, dict)

                # Args should be all fake tensors or all real tensors
                if _can_all_args_be_materialized(args):
                    args = tree_map(materialize, args)
                if _can_all_args_be_materialized(kwargs):
                    kwargs = tree_map(materialize, kwargs)
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

            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)][self.warmup:])
            wall_time = statistics.mean(walltimes[self.warmup:]) * 1000
            with maybe_disable_fake_tensor_mode():
                vals_to_bcast = torch.tensor([device_time, wall_time], device=device)
                dist.broadcast(vals_to_bcast, 0)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()

            n.meta["device_time"] = statistics.mean([s.elapsed_time(e)
                                                     for s, e in zip(start_events, end_events)][self.warmup:])
            n.meta["wall_time"] = statistics.mean(walltimes[self.warmup:]) * 1000

        return fake_ret
