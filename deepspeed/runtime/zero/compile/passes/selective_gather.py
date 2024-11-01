# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Tuple

import torch
from torch.fx import Graph

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from ..stage3_backend import param_manager, profiling_results

max_alloc_mem = 0
last_optimize_step = 0
nz3 = None


def selective_gather(graph: Graph, graph_id: int, mem: List[Tuple[str, int, int]], op_time: List[Tuple[str, int, int]],
                     tensor_sizes: List[Tuple[str, int]], mem_budget: float, bwd: bool) -> Graph:

    print(f"selective_gather starting graph_id={graph_id}")
    return graph

    # ds_id_to_size = {}
    # ds_id_to_time = defaultdict(float)
    # ds_id_to_prof_dtime = defaultdict(float)
    # ds_id_to_prof_wtime = defaultdict(float)

    # for graph_id, pm in param_manager.items():
    #     params = pm.params
    #     for param_name, param in params.items():
    #         ds_id = pm.ds_ids[param_name]
    #         ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

    #     profile = profiling_results[graph_id]
    #     for n in profile.fwd_graph.nodes:
    #         if n.target == torch.ops.native_z3.allgather_param:
    #             assert "tensor_size" in n.meta
    #             ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
    #             assert "device_time" in n.meta
    #             ds_id_to_time[n.args[2]] += n.meta["device_time"]

    #             ds_id_to_prof_dtime[n.args[2]] = n.meta["device_time"]
    #             ds_id_to_prof_wtime[n.args[2]] = n.meta["wall_time"]

    #     if profile.bwd_graph is not None:
    #         for n in profile.bwd_graph.nodes:
    #             if n.target == torch.ops.native_z3.allgather_param:
    #                 assert "tensor_size" in n.meta
    #                 ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
    #                 assert "device_time" in n.meta
    #                 ds_id_to_time[n.args[2]] += n.meta["device_time"]

    # ds_ids = list(ds_id_to_size.keys())
    # ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    # # print(f"ds_id_to_size={ds_id_to_size}")
    # # print(f"ds_id_to_time={ds_id_to_time}")

    # if dist.get_rank() == 0:
    #     for ds_id in ds_ids:
    #         dtime_in_sec = ds_id_to_prof_dtime[ds_id]
    #         wtime_in_sec = ds_id_to_prof_wtime[ds_id]
    #         size_in_mb = ds_id_to_size[ds_id] / 1024 / 1024
    #         print(
    #             f"ds_id={ds_id} time_per_size={ds_id_to_time[ds_id] / ds_id_to_size[ds_id]:.5f} dtime={dtime_in_sec:.3f} wtime={wtime_in_sec:.3f} size={size_in_mb:.2f}MB bw={size_in_mb/dtime_in_sec:.2f}MB/s"
    #         )

    # sorted_ds_ids = {ds_id: ds_id_to_size[ds_id] for ds_id in ds_ids}

    # accelerator = get_accelerator()
    # max_alloc_mem = accelerator.max_memory_allocated()
    # total_mem = accelerator.total_memory()
    # available_mem = (total_mem - max_alloc_mem) - MEM_MARGIN

    # persistent_mem = 0
    # for ds_id, size in sorted_ds_ids.items():
    #     if persistent_mem + size > available_mem:
    #         break
    #     persistent_mem += size
    #     nz3.set_persistent(ds_id, True)
    #     if dist.get_rank() == 0:
    #         print(f"Set persistent: {ds_id} size: {size} persistent_mem: {persistent_mem}")
