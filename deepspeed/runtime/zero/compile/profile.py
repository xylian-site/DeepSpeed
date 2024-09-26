# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import types
from typing import Any
import statistics

import torch
from torch.utils._pytree import tree_all
from torch.fx import GraphModule, Interpreter

from torch._subclasses.fake_tensor import is_fake
try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # torch < v2.5
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode as unset_fake_temporarily

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from .util import is_comm_op


def _all_real_if_tensor(args):
    return tree_all(lambda x: not torch.is_tensor(x) or not is_fake(x), args)


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, nz3: types.ModuleType, gm: GraphModule, iteration: int = 10, warmup: int = 5):
        super().__init__(gm)

        self.nz3 = nz3

        assert iteration > 0
        assert warmup >= 0
        assert warmup < iteration
        self.iteration = iteration
        self.warmup = warmup

    def run(self, *args) -> Any:
        """Run the graph with profiling enabled.

        args: inputs to the graph. Tensors in the inpusts must be real tensors, not fake tensors. args can contain ds parameters.
        returns: The output of the graph. Tensor in the output is real tensors.
        """

        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[get_accelerator().current_device_name()]):
                    return_val = super().run(*args)
        except Exception as e:
            print(f"Profiling error {e}")
        finally:
            self.nz3.enable_profiling(False)
        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:

        if n.op in {"placeholder", "output"}:
            n.meta["device_time"] = 0.0
            n.meta["wall_time"] = 0.0
            return super().run_node(n)
        else:
            accelerator = get_accelerator()
            device = torch.device(accelerator.current_device())
            start_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]
            end_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]

            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            if is_comm_op(n):
                dist.barrier()

            walltimes = []
            for i in range(self.iteration):
                start = time.time()
                start_events[i].record()
                out = getattr(self, n.op)(n.target, args, kwargs)
                end_events[i].record()
                walltimes.append(time.time() - start)

            if is_comm_op(n):
                dist.barrier()

            accelerator.synchronize()

            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)][self.warmup:])
            wall_time = statistics.mean(walltimes[self.warmup:]) * 1000

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time], device=device)
                dist.broadcast(vals_to_bcast, 0)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()

        return out
