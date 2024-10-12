# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import types
from typing import Any, Tuple
import statistics
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_all
from torch.fx import GraphModule, Interpreter
from torch.fx.node import map_aggregate

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


def _to(v, device):
    if torch.is_tensor(v):
        with unset_fake_temporarily():
            v.data = v.to(device)
    return v


def _args_to_key(v):

    def _tensor_to_key(v) -> str:
        if torch.is_tensor(v):
            if v.numel() == 1:
                return f"{v.dtype}{v.device}{v.item()}"
            else:
                return f"{v.dtype}{v.device}{v.shape}"
        return str(v)

    return map_aggregate(v, _tensor_to_key)


@dataclass
class AllGatheredParamWrapperValue:
    sliced_param: torch.Tensor
    meta_param: torch.Tensor
    device: torch.device
    rank: int
    world_size: int
    padded_size: int


def _create_allgathered_param_wrapper(param: torch.Tensor, rank: int, world_size: int) -> AllGatheredParamWrapperValue:
    # Create a tensor with the same size as the original tensor
    meta_param = param.to("meta")

    padded_size = (param.numel() + world_size - 1) // world_size * world_size
    slice_size = padded_size // world_size
    start_idx = min(rank * slice_size, param.numel())
    end_idx = min((rank + 1) * slice_size, param.numel())
    sliced_param = torch.empty(slice_size, dtype=param.dtype, device=param.device)
    if start_idx < end_idx:
        sliced_param.copy_(param.view(-1).narrow(0, start_idx, end_idx - start_idx))

    return AllGatheredParamWrapperValue(sliced_param, meta_param, param.device, rank, world_size, padded_size)


def _rebuild_param(allgathered_param: AllGatheredParamWrapperValue) -> torch.Tensor:
    buffer = torch.empty([allgathered_param.padded_size],
                         dtype=allgathered_param.sliced_param.dtype,
                         device=allgathered_param.device)
    dist.all_gather_into_tensor(buffer, allgathered_param.sliced_param)
    return buffer.narrow(0, 0, allgathered_param.meta_param.numel()).view(allgathered_param.meta_param.shape)


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
        self.device = torch.device(get_accelerator().current_device())
        self.cache: dict[Tuple, Any] = {}
        self.distributed = dist.is_initialized()

    def run(self, *args) -> Any:
        """Run the graph with profiling enabled.

        args: inputs to the graph. Tensors in the inpusts must be real tensors, not fake tensors. args can contain ds parameters.
        returns: The output of the graph. Tensor in the output is real tensors.
        """
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
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

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        def rebuild_param_if_necessary(v):
            if isinstance(v, AllGatheredParamWrapperValue):
                return _rebuild_param(v)
            return v

        args = map_aggregate(args, lambda x: rebuild_param_if_necessary(x))

        args = map_aggregate(args, lambda x: _to(x, self.device))
        kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))

        cache_key = (n.target, _args_to_key(args), _args_to_key(kwargs))
        cache_hit = cache_key in self.cache
        if cache_hit:
            device_time, wall_time = self.cache[cache_key]
            n.meta["device_time"] = device_time
            n.meta["wall_time"] = wall_time

        if is_comm_op(n):
            assert self.distributed, f"Distributed environment is not initialized but comm operator {n.name} {n.target} is used."
            dist.barrier()

        run_only_once = cache_hit or n.target == torch.ops.native_z3.release_param
        iteration = 1 if run_only_once else self.iteration
        accelerator = get_accelerator()
        start_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]
        end_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]

        walltimes = []
        for i in range(iteration):
            start = time.time()
            start_events[i].record()
            out = getattr(self, n.op)(n.target, args, kwargs)
            end_events[i].record()
            walltimes.append(time.time() - start)

        if is_comm_op(n):
            dist.barrier()

        accelerator.synchronize()

        if not cache_hit:
            warmup = 0 if run_only_once else self.warmup
            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)][warmup:])
            wall_time = statistics.mean(walltimes[warmup:]) * 1000

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time], device=self.device)
                if self.distributed:
                    dist.all_reduce(vals_to_bcast, dist.ReduceOp.AVG)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()
                self.cache[cache_key] = (n.meta["device_time"], n.meta["wall_time"])

        if n.target == torch.ops.native_z3.allgather_param:
            self.nz3.invalidate_gathered_param(args[1])
            out = _create_allgathered_param_wrapper(out, dist.get_rank(), dist.get_world_size())

        return out
