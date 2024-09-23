# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import types
from typing import Any, Dict
import statistics

import torch
from torch.fx import GraphModule, Interpreter
from torch.fx.node import map_aggregate, Argument

from torch._subclasses.fake_tensor import is_fake
try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # torch < v2.5
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode as unset_fake_temporarily

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from .util import is_comm_op


def _materialize_meta(t):
    if torch.is_tensor(t) and is_fake(t):
        return torch.zeros(t.shape, dtype=t.dtype, layout=t.layout, device=t.device)
    return t


def _load_value(env, v):
    if isinstance(v, torch.fx.Node):
        return env[v]
    return v


def _materialize_args_kwargs(args, kwargs, env):
    args = map_aggregate(args, lambda x: _materialize_meta(x))
    kwargs = map_aggregate(kwargs, lambda x: _materialize_meta(x))
    return args, kwargs


def _dematerialize(out):
    # This runs under fake mode
    def _dematerialize_tensor(t):
        if torch.is_tensor(t):
            return torch.empty(t.shape, dtype=t.dtype, layout=t.layout, device=t.device)
        return t

    return map_aggregate(out, _dematerialize_tensor)


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, nz3: types.ModuleType, gm: GraphModule, iteration: int = 10, warmup: int = 5):
        super().__init__(gm)

        self.nz3 = nz3
        self.env_values: Dict[torch.fx.Node, Argument] = {}

        assert iteration > 0
        assert warmup >= 0
        assert warmup < iteration
        self.iteration = iteration
        self.warmup = warmup

    def run(self, *args) -> Any:
        try:
            self.nz3.enable_profiling(True)
            return_val = super().run(*args)
        except Exception as e:
            print(f"Profiling error {e}")
        finally:
            self.nz3.enable_profiling(False)
        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:

        # print(f"Profiling starting {n.name} {n.op} {n.target}")

        n.meta["device_time"] = 0.0
        n.meta["wall_time"] = 0.0

        # Run in fake mode
        fake_ret = super().run_node(n)

        if n.op in {"placeholder", "output"}:
            self.env_values[n] = fake_ret
        else:
            accelerator = get_accelerator()
            start_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]
            end_events = [accelerator.Event(enable_timing=True) for _ in range(self.iteration)]

            out = None
            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[get_accelerator().current_device_name()]):
                    args, kwargs = self.fetch_args_kwargs_from_env(n)
                    assert isinstance(args, tuple)
                    assert isinstance(kwargs, dict)

                    args, kwargs = _materialize_args_kwargs(args, kwargs, self.env_values)

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

            # Save fake value using result from non-fake mode.
            # This is necessary to create backward inputs that don't have symbolic dimensions.
            self.env_values[n] = _dematerialize(out)

            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)][self.warmup:])
            wall_time = statistics.mean(walltimes[self.warmup:]) * 1000

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time],
                                             device=torch.device(accelerator.current_device()))
                dist.broadcast(vals_to_bcast, 0)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()

        return fake_ret
