# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

from torch.fx import Graph, Node

from .graph_param import DSGraphParamManager
from .util import tensor_meta_size


def get_original_args_num(node: Node):
    if node.name.startswith("allgather_ds_param") \
        or node.name.startswith("release_ds_param") \
        or node.name.startswith("wait_allgather_ds_param") \
        or node.name.startswith("reduce_ds_param"):
        return 1

    return len(node.args)


def filter_args(node: Node):
    args = node.args[:get_original_args_num(node)]
    return [arg for arg in args if isinstance(arg, Node)]


def get_runnable_nodes(scheduled: List[Node], unscheduled: List[Node]):
    scheduled = set(scheduled)
    return [node for node in unscheduled if all(arg in scheduled for arg in filter_args(node))]


def choose_next_node(scheduled: List[Node], unscheduled: List[Node]):
    runnable_nodes = get_runnable_nodes(scheduled, unscheduled)
    return runnable_nodes[0]


def create_mem_table(graph: Graph) -> dict:
    mem_table = {}
    for node in graph.nodes:
        if node.name.startswith("allgather_ds_param"):
            mem_table[node.name] = tensor_meta_size(node.meta["tensor_meta"])
        elif node.name.startswith("release_ds_param") or node.name.startswith("reduce_ds_param"):
            mem_table[node.name] = -tensor_meta_size(node.meta["tensor_meta"])
        else:
            mem_table[node.name] = 0

    return mem_table


def list_schedule(graph: Graph, param_manager: DSGraphParamManager, bwd: bool) -> List[Node]:

    mem_table = create_mem_table(graph)

    scheduled = []
    unscheduled = []
    for node in graph.nodes:
        # print(f"Node: {node} args: {node.args}")
        if len(node.args) == 0:
            scheduled.append(node)
        else:
            unscheduled.append(node)

    while len(unscheduled) > 0:
        next_node = choose_next_node(scheduled, unscheduled)
        print(f"Next node: {next_node}")
        scheduled.append(next_node)
        unscheduled.remove(next_node)

    # Placeholder for the list scheduler
    # return param_manager.param_nodes
