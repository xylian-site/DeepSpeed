# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import Any, Dict, List

import torch
from torch.utils._pytree import tree_map
from torch.fx import Node, GraphModule, Interpreter

from deepspeed.accelerator import get_accelerator


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, iteration: int = 10):
        super().__init__(gm)
        self.runtimes_ms: Dict[Node, List[float]] = defaultdict(list)
        self.iteration = iteration

    def run_node(self, n: torch.fx.Node) -> Any:
        fake_ret = super().run_node(n)

        if n.op in {"placeholder", "output"}:
            self.runtimes_ms[n].append(0)
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

            for i in range(self.iteration):
                start_events[i].record()
                getattr(self, n.op)(n.target, args, kwargs)
                end_events[i].record()

            accelerator.synchronize()
            self.runtimes_ms[n] = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # print(f"Node {n} took {self.runtimes_ms[n]} ms to run")
        return fake_ret
