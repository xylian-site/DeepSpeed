# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import List

import torch
from torch.fx import Graph

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # torch < v2.5
    from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode as unset_fake_temporarily

from ..profilers import ProfilingResult
from ..graph_param import DSGraphParamManager
from ..fx import move_primals_to_head

MARGIN = 0.1

copy_stream = None
offload_event = None
reload_event = None

offload_key_events = {}
reload_key_events = {}


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


def move_key(state, key, key_event=None):
    offload_buf_key = _make_offload_state_key(key)
    if offload_buf_key not in state:
        state[offload_buf_key] = get_accelerator().pin_memory(torch.empty_like(state[key], device="cpu"))

    with get_accelerator().stream(copy_stream):
        state[offload_buf_key].copy_(state[key], non_blocking=True)
        state[key].record_stream(copy_stream)

    if key_event is None:
        offload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def move_back_key(state, key, key_event=None):
    with get_accelerator().stream(copy_stream):
        state[key] = state[_make_offload_state_key(key)].to(device, non_blocking=True)

    if key_event is None:
        reload_event.record(stream=copy_stream)
    else:
        key_event.record(stream=copy_stream)


def offload_adam_states_async():

    print("Offloading Adam states")
    for i, (k, state) in enumerate(optimizer.state.items()):
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

    print("Reloading Adam states")

    for _, state in optimizer.state.items():
        if _make_offload_state_key("exp_avg") in state:
            move_back_key(state, "exp_avg")
        if _make_offload_state_key("exp_avg_sq") in state:
            move_back_key(state, "exp_avg_sq")

    get_accelerator().synchronize()


def sync_offload_states(event=None):
    if event is None:
        offload_event.wait(copy_stream)
    else:
        event.wait(copy_stream)


def sync_reload_states(event=None):
    if event is None:
        reload_event.wait(copy_stream)
    else:
        event.wait(copy_stream)


def make_offload_task(task):

    def run_offload_task():
        state = optimizer.state[task[1]]
        if offload_key_events.get(task[1]) is None:
            offload_key_events[task[1]] = get_accelerator().Event()
        print(f"run_offload_task {task[0]} {task[2]} {task[3]} {task[4]}")
        move_key(state, task[2], offload_key_events[task[1]])

    return run_offload_task


def make_offload_sync(task):

    def run_offload_sync():
        event = offload_key_events[task[1]]
        sync_offload_states(event)

    return run_offload_sync


def make_reload_task(task):

    def run_reload_task():
        state = optimizer.state[task[1]]
        if reload_key_events.get(task[1]) is None:
            reload_key_events[task[1]] = get_accelerator().Event()
        print(f"run_reload_task {task[0]} {task[2]} {task[3]} {task[4]}")
        move_back_key(state, task[2], reload_key_events[task[1]])

    return run_reload_task


offload_tasks = []
offload_tasks_remaining = []
reload_task_remaining = []


def offload_opt_states_inc(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                           mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:

    to_remove = []
    for node in graph.nodes:
        if node.op == 'call_function' and \
            node.target in [offload_adam_states_async, sync_offload_states, reload_adam_states_async, sync_reload_states]:
            to_remove.append(node)

    for node in to_remove:
        graph.erase_node(node)

    accelerator = get_accelerator()
    total_mem = accelerator.total_memory() * (1 - MARGIN)

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    mem_dict = {name: peak for name, alloc_mem, delta, peak in mem}

    current_peak_mem = 0
    peak_mem = {}
    for node in graph.nodes:
        # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
        if mem_dict[node.name] > current_peak_mem:
            current_peak_mem = mem_dict[node.name]
        peak_mem[node.name] = current_peak_mem

    # fwd_max_mem = max(m[3] for m in prof.fwd_mem)
    # bwd_max_mem = max(m[3] for m in prof.bwd_mem) if len(prof.bwd_mem) > 0 else 0
    # peak_mem = max(peak_mem, fwd_max_mem, bwd_max_mem)

    global offload_tasks_remaining, reload_tasks_remaining

    # print(f"offload_opt_states_inc bwd={bwd}")
    if not bwd:
        is_first_graph = graph_id == graph_order[0][0]
        print(f"offload_opt_states_inc graph {graph_id} graph_order {graph_order} fwd is_first_graph {is_first_graph}")

        # At the beginning of the first graph, we schedule offload tasks to launch all offloading
        if is_first_graph:

            with unset_fake_temporarily():
                reload_adam_states_async()
                sync_reload_states()

            for i, (k, state) in enumerate(optimizer.state.items()):
                if _make_offload_state_key("exp_avg") in state:
                    key = _make_offload_state_key("exp_avg")
                    offload_tasks.append(
                        (i, k, "exp_avg", state[key].numel() * state[key].element_size(), state[key].dtype))
                if _make_offload_state_key("exp_avg_sq") in state:
                    key = _make_offload_state_key("exp_avg_sq")
                    offload_tasks.append(
                        (i, k, "exp_avg_sq", state[key].numel() * state[key].element_size(), state[key].dtype))

            for t in offload_tasks:
                print(f"Offloading task {t[0]} {t[2]} {t[3]}")

            inserted_offload = False
            for node in graph.nodes:
                # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
                if node.op != 'placeholder' and not inserted_offload:
                    # print(f"Inserting offload_opt before {node.name}")
                    for task in offload_tasks:
                        name = f"offload_opt_{task[0]}_{task[2]}"
                        with graph.inserting_before(node):
                            offload_node = graph.create_node('call_function',
                                                             make_offload_task(task), (), {},
                                                             name=name)
                    inserted_offload = True

            offload_tasks_remaining = copy.copy(offload_tasks)

        prev_node = None
        for node in graph.nodes:

            if node.name not in peak_mem:
                continue

            to_offload = []
            optim_size = sum([task[3] for task in offload_tasks_remaining])
            while total_mem - peak_mem[node.name] - optim_size < 0:
                if len(offload_tasks_remaining) == 0:
                    break

                task = offload_tasks_remaining.pop(0)
                to_offload.append(task)
                optim_size = sum([task[3] for task in offload_tasks_remaining])

            for task in to_offload:
                with graph.inserting_before(node):
                    graph.create_node('call_function',
                                      make_offload_sync(task), (), {},
                                      name=f"offload_opt_sync_{task[0]}_{task[2]}")

        print(f"offload_opt_states_inc graph {graph_id} fwd graph {graph}")

    else:
        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_first_graph = graph_id == graph_order_with_backward[-1]
        is_last_graph = graph_id == graph_order_with_backward[0]

        if is_first_graph:
            inserted_sync = False
            for node in graph.nodes:
                if node.op != 'placeholder' and not inserted_sync:
                    # print(f"Inserting offload_opt before {node.name}")
                    for task in offload_tasks:
                        name = f"offload_opt_sync_{task[0]}_{task[2]}"
                        with graph.inserting_before(node):
                            graph.create_node('call_function', make_offload_sync(task), (), {}, name=name)
                    inserted_sync = True
            reload_tasks_remaining = copy.copy(offload_tasks)

        for node in graph.nodes:
            if node.name not in peak_mem:
                continue

            if len(reload_tasks_remaining) > 0:
                task = reload_tasks_remaining[0]

                while total_mem - peak_mem[node.name] - task[3] > 0:
                    reload_tasks_remaining.pop(0)
                    with graph.inserting_after(node):
                        graph.create_node('call_function',
                                          make_reload_task(task), (), {},
                                          name=f"reload_opt_{task[0]}_{task[2]}")
                    if len(reload_tasks_remaining) == 0:
                        break
                    task = reload_tasks_remaining[0]

        if is_last_graph:
            for node in graph.nodes:
                # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
                if node.op == 'output':
                    sync_fn = lambda: copy_stream.synchronize()
                    with graph.inserting_before(node):
                        graph.create_node('call_function', sync_fn, (), {}, name="sync_offload_copy_stream")

        print(
            f"offload_opt_states_inc graph {graph_id} graph_order {graph_order} bwd is_first_graph {is_first_graph} is_last_graph {is_last_graph} {graph}"
        )

    return graph


def insert_offload_opt_states(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                              mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:

    if bwd:
        graph_order_with_backward = [g[0] for g in graph_order if g[1]]
        is_last_graph = graph_id == graph_order_with_backward[0]

        if not is_last_graph:
            return graph

        inserted_reload = False
        for node in graph.nodes:
            # print(f"Node: {node.name} mem: {mem_dict[node.name]}")
            if node.op == 'output' and not inserted_reload and is_last_graph:
                # print(f"Inserting reload_opt before {node.name}")
                with graph.inserting_before(node):
                    reload_node = graph.create_node('call_function',
                                                    reload_adam_states_async, (), {},
                                                    name="reload_opt")
                with graph.inserting_after(reload_node):
                    graph.create_node('call_function', sync_reload_states, (), {}, name="sync_reload_opt")
                inserted_reload = True
    else:
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


def move_offload_opt_states(graph: Graph, graph_id: int, graph_order: List[int], profiling_results: ProfilingResult,
                            mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> Graph:
    return offload_opt_states_inc(graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd)


def init_offload_opt_states(adam_optimizer):
    lazy_init()

    global optimizer
    optimizer = adam_optimizer
    global device
    device = torch.device(get_accelerator().current_device())
