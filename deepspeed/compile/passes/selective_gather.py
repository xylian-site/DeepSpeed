# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List

import torch
from torch.fx import GraphModule, Graph

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..util import get_deepcompile_handle
from ..graph_param import DSGraphParamManager

NAME = "selective_gather"

SIZE_THRESHOLD = 64e6

max_alloc_mem = 0
last_optimize_step = 0
idle_times = defaultdict(float)


def _is_comm_node(node):
    if not isinstance(node, torch.fx.Node):
        return False
    return node.target == torch.ops.dc.allgather_param.default or node.target == torch.ops.dc.reduce_grad.default


def _is_allgather_node(node):
    if not isinstance(node, torch.fx.Node):
        return False
    return node.target == torch.ops.dc.allgather_param.default


def simulate_time(graph: Graph):
    global idle_times

    t_comp = 0
    t_comm = 0

    last_comm_node = None

    for n in graph.nodes:
        if "device_time" not in n.meta:
            continue

        ready_time = 0
        for arg in n.args:
            if _is_comm_node(arg):
                ready_time = max(ready_time, t_comm)
            else:
                ready_time = max(ready_time, t_comp)

        device_time = n.meta["device_time"] if "device_time" in n.meta else 0
        if _is_comm_node(n):
            t_comm = device_time + ready_time
            last_comm_node = n
        else:
            # Waiting for communication to finish
            if ready_time > t_comp and _is_allgather_node(last_comm_node):
                ds_id = last_comm_node.args[2]
                idle_times[ds_id] = ready_time - t_comp
                print(f"setting idle time for {last_comm_node.name} (ds_id={ds_id}) to {idle_times[ds_id]}")
            t_comp = device_time + ready_time


def selective_gather(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results, create_inputs_fn,
                     mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:

    simulate_time(gm.graph)

    if not bwd:
        return gm

    last_backward_graph_id = None
    for g_id, needs_bwd in graph_order:
        if needs_bwd:
            last_backward_graph_id = g_id
            break

    # Run only on the last backward graph
    if last_backward_graph_id is None or graph_id != last_backward_graph_id:
        return gm

    peak_mem = 0
    for graph_id, prof in profiling_results.items():
        # Use peak memory
        fwd_max_mem = max(m[3] for m in prof.fwd_mem)
        bwd_max_mem = max(m[3] for m in prof.bwd_mem) if len(prof.bwd_mem) > 0 else 0
        peak_mem = max(peak_mem, fwd_max_mem, bwd_max_mem)
        if dist.get_rank() == 0:
            print(
                f"selective_gather graph_id={graph_id} max_mem={peak_mem} fwd_max_mem={fwd_max_mem} bwd_max_mem={bwd_max_mem}"
            )

    persistent_ds_ids = set()
    for graph_id, pm in param_manager.items():
        for name, ds_param in pm.params.items():
            if ds_param.param.ds_persist:
                persistent_ds_ids.add(pm.ds_ids[name])

    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)
    ds_id_to_prof_dtime = defaultdict(float)
    ds_id_to_prof_wtime = defaultdict(float)

    for graph_id, pm in param_manager.items():
        params = pm.params
        for param_name, param in params.items():
            ds_id = pm.ds_ids[param_name]
            ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

        profile = profiling_results[graph_id]
        for n in profile.fwd_graph.nodes:
            if n.target == torch.ops.dc.allgather_param.default:
                assert "tensor_size" in n.meta
                ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                assert "device_time" in n.meta
                ds_id_to_time[n.args[2]] += n.meta["device_time"]

                ds_id_to_prof_dtime[n.args[2]] = n.meta["device_time"]
                ds_id_to_prof_wtime[n.args[2]] = n.meta["wall_time"]

        if profile.bwd_graph is not None:
            for n in profile.bwd_graph.nodes:
                if n.target == torch.ops.dc.allgather_param.default:
                    assert "tensor_size" in n.meta
                    ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                    assert "device_time" in n.meta
                    ds_id_to_time[n.args[2]] += n.meta["device_time"]

    ds_ids = [ds_id for ds_id in ds_id_to_size if ds_id not in persistent_ds_ids]
    ds_ids.sort(key=lambda ds_id: ds_id_to_size[ds_id])

    # 1. Put params smaller than SIZE_THRESHOLD
    target_ds_ids = []
    for ds_id in ds_ids:
        size = ds_id_to_size[ds_id]
        if size < SIZE_THRESHOLD:
            target_ds_ids.append(ds_id)

    # 2. Put params that cause long idle time
    sorted_idle_times = sorted(idle_times.items(), key=lambda x: x[1], reverse=True)
    for ds_id, idle_time in sorted_idle_times:
        if ds_id in target_ds_ids:
            continue
        target_ds_ids.append(ds_id)

    accelerator = get_accelerator()
    total_mem = accelerator.total_memory()
    vals_to_bcast = torch.tensor([total_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    total_mem = vals_to_bcast[0].item()

    MEM_MARGIN = 0.1
    available_mem = total_mem * (1 - MEM_MARGIN) - peak_mem

    if dist.get_rank() == 0:
        print(
            f"selective_gather max_mem={peak_mem} total_mem={total_mem} MEM_MARGIN={MEM_MARGIN} available_mem={available_mem}"
        )

    ds_id_to_param = {}
    for g_id, g_pm in param_manager.items():
        for name, ds_param in g_pm.params.items():
            ds_id_to_param[g_pm.ds_ids[name]] = ds_param.param

    persistent_mem = 0
    nz3 = get_deepcompile_handle()
    for ds_id in target_ds_ids:
        size = ds_id_to_size[ds_id]
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size

        param_obj = ds_id_to_param[ds_id]

        nz3.set_persistent(ds_id)
        if dist.get_rank() == 0:
            print(f"Set persistent: {ds_id} size: {size} persistent_mem: {persistent_mem} shape: {param_obj.ds_shape}")

    return gm
