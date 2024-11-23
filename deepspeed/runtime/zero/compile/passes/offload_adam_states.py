# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import torch
from torch.fx import Graph

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key

from ..profilers import ProfilingResult
from ..graph_param import DSGraphParamManager
from ..fx import move_primals_to_head

MARGIN = 0.1

copy_stream = None
offload_event = None
reload_event = None


def lazy_init():
    global copy_stream
    global offload_event
    global reload_event

    if copy_stream is None:

        copy_stream = get_accelerator().Stream()
        offload_event = get_accelerator().Event()
        reload_event = get_accelerator().Event()


optimizer = None
device = None


def offload_adam_states_async():

    print("Offloading Adam states")

    def move_key(state, key):
        offload_buf_key = _make_offload_state_key(key)
        if offload_buf_key not in state:
            state[offload_buf_key] = get_accelerator().pin_memory(torch.empty_like(state[key], device="cpu"))

        with get_accelerator().stream(copy_stream):
            state[offload_buf_key].copy_(state[key], non_blocking=True)
            state[key].record_stream(copy_stream)

        offload_event.record(stream=copy_stream)

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_key(state, "exp_avg_sq")

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            del state["exp_avg"]
        if "exp_avg_sq" in state:
            del state["exp_avg_sq"]

    get_accelerator().synchronize()


def reload_adam_states_async():

    def move_back_key(state, key):
        with get_accelerator().stream(copy_stream):
            state[key] = state[_make_offload_state_key(key)].to(device, non_blocking=True)
        reload_event.record(stream=copy_stream)

    for _, state in optimizer.state.items():
        if _make_offload_state_key("exp_avg") in state:
            move_back_key(state, "exp_avg")
        if _make_offload_state_key("exp_avg_sq") in state:
            move_back_key(state, "exp_avg_sq")

    get_accelerator().synchronize()


def sync_offload_states():
    offload_event.wait_stream(copy_stream)


def sync_reload_states():
    reload_event.wait_stream(copy_stream)


def offload_opt_states_fwd(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                           mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:

    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, delta) for name, alloc_mem, delta in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    total_param_size = sum(
        [tensor_size_dict[n.name] for n in graph.nodes if n.target == torch.ops.native_z3.allgather_param])

    pm = param_manager[graph_id]

    is_first_graph = graph_id == graph_order[0][0]

    graph = move_primals_to_head(graph)

    inserted_offload = False
    for node in graph.nodes:
        # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
        if node.op != 'placeholder' and not inserted_offload and is_first_graph:
            # print(f"Inserting offload_opt before {node.name}")
            with graph.inserting_before(node):
                offload_node = graph.create_node('call_function',
                                                 offload_adam_states_async, (), {},
                                                 name="offload_opt")
            with graph.inserting_after(offload_node):
                graph.create_node('call_function', sync_offload_states, (), {}, name="sync_offload_opt")

            inserted_offload = True

    return graph


def offload_opt_states_bwd(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                           mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:
    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, delta) for name, alloc_mem, delta in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    total_param_size = sum(
        [tensor_size_dict[n.name] for n in graph.nodes if n.target == torch.ops.native_z3.allgather_param])

    pm = param_manager[graph_id]

    graph_order_with_backward = [g[0] for g in graph_order if g[1]]
    is_last_graph = graph_id == graph_order_with_backward[0]

    inserted_reload = False
    for node in graph.nodes:
        # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
        if node.op == 'output' and not inserted_reload and is_last_graph:
            # print(f"Inserting reload_opt before {node.name}")
            with graph.inserting_before(node):
                reload_node = graph.create_node('call_function', reload_adam_states_async, (), {}, name="reload_opt")
            with graph.inserting_after(reload_node):
                graph.create_node('call_function', sync_reload_states, (), {}, name="sync_reload_opt")
            inserted_reload = True

    return graph


def offload_opt_states(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                       mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:
    lazy_init()

    if bwd:
        return offload_opt_states_bwd(graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd)
    else:
        return offload_opt_states_fwd(graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd)


def init_offload_opt_states(adam_optimizer):
    lazy_init()

    global optimizer
    optimizer = adam_optimizer
    global device
    device = torch.device(get_accelerator().current_device())
