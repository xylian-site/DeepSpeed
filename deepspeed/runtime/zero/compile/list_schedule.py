# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Dict
from copy import copy

from torch.fx import Graph, Node
from torch.utils._pytree import tree_iter

from .util import tensor_meta_size


def init_schdule(graph: Graph):
    mem_table = create_mem_table(graph)

    scheduled = []
    unscheduled = []
    for node in graph.nodes:
        # print(f"Node: {node} args: {node.args}")
        if len(node.args) == 0:
            scheduled.append(node)
        else:
            unscheduled.append(node)

    return scheduled, unscheduled, mem_table


def make_graph_from_schedule(scheduled: List[Node]):
    new_graph = Graph()
    env = {}
    for node in scheduled:
        new_node = new_graph.node_copy(node, lambda n: env[n.name])
        env[node.name] = new_node

    return new_graph


def get_original_args_num(node: Node):
    if node.name.startswith("allgather_ds_param") \
        or node.name.startswith("release_ds_param") \
        or node.name.startswith("wait_allgather_ds_param") \
        or node.name.startswith("reduce_ds_param"):
        return 1

    return len(node.args)


def flat_nodes_in_args(args: List[Node]):
    return [a for a in tree_iter(args) if isinstance(a, Node)]


def filter_args(node: Node):
    args = node.args[:get_original_args_num(node)]
    return flat_nodes_in_args(args)


def get_runnable_nodes(scheduled: List[Node], unscheduled: List[Node]):
    scheduled = set(scheduled)
    return [node for node in unscheduled if all(arg in scheduled for arg in filter_args(node))]


def choose_next_node(scheduled: List[Node], unscheduled: List[Node], mem_table: Dict[str, int]):
    runnable_nodes = get_runnable_nodes(scheduled, unscheduled)

    # sort by memory usage
    runnable_nodes = sorted(runnable_nodes, key=lambda n: mem_table[n.name])
    return runnable_nodes[0]


def create_mem_table(graph: Graph) -> Dict[str, int]:
    mem_table = {}
    for node in graph.nodes:
        if node.name.startswith("allgather_ds_param"):
            mem_table[node.name] = tensor_meta_size(node.meta["tensor_meta"])
        elif node.name.startswith("release_ds_param") or node.name.startswith("reduce_ds_param"):
            mem_table[node.name] = -tensor_meta_size(node.meta["tensor_meta"])
        else:
            mem_table[node.name] = 0

    return mem_table


def list_schedule(graph: Graph) -> Graph:

    scheduled, unscheduled, mem_table = init_schdule(graph)

    while len(unscheduled) > 0:
        next_node = choose_next_node(scheduled, unscheduled, mem_table)
        scheduled.append(next_node)
        unscheduled.remove(next_node)

    return make_graph_from_schedule(scheduled)


###############################


def schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node]):
    runnable = get_runnable_nodes(scheduled, unscheduled)
    non_ag_runnable = [n for n in runnable if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)

    while len(non_ag_runnable) > 0:
        next_node = non_ag_runnable[0]
        tmp_scheduled.append(next_node)
        tmp_unscheduled.remove(next_node)

        runnable = get_runnable_nodes(tmp_scheduled, tmp_unscheduled)
        non_ag_runnable = [n for n in runnable if not n.name.startswith("allgather_ds_param")]

    return tmp_scheduled, tmp_unscheduled


def list_schedule2(graph: Graph) -> Graph:

    scheduled, unscheduled, mem_table = init_schdule(graph)

    tmp_scheduled, tmp_unscheduled = schedule_without_allgather(scheduled, unscheduled)

    while len(tmp_unscheduled) > 0:

        runnable = get_runnable_nodes(tmp_scheduled, tmp_unscheduled)

        ag_with_unblock_time = []
        for ag_node in runnable:
            ag_scheduled = copy(tmp_scheduled)
            ag_unscheduled = copy(tmp_unscheduled)
            ag_scheduled.append(ag_node)
            ag_unscheduled.remove(ag_node)
            ag_scheduled, ag_unscheduled = schedule_without_allgather(ag_scheduled, ag_unscheduled)
            unblock_time = sum(n.meta["device_time"] for n in ag_scheduled[len(tmp_scheduled) + 1:])
            ag_with_unblock_time.append((ag_node, unblock_time))

        ag_with_unblock_time = sorted(ag_with_unblock_time, key=lambda x: x[1], reverse=True)
        best_ag_node = ag_with_unblock_time[0][0]

        new_scheduled_without_allgahter = tmp_scheduled[len(scheduled):]
        scheduled.append(best_ag_node)
        unscheduled.remove(best_ag_node)
        for n in new_scheduled_without_allgahter:
            scheduled.append(n)
            unscheduled.remove(n)

        tmp_scheduled, tmp_unscheduled = schedule_without_allgather(scheduled, unscheduled)

    return make_graph_from_schedule(tmp_scheduled)
