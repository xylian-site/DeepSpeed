# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Dict

from torch.fx import Graph, Node
from torch.utils._pytree import tree_iter

from .util import tensor_meta_size


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
        next_node = choose_next_node(scheduled, unscheduled, mem_table)
        print(f"Next node: {next_node} mem: {mem_table[next_node.name]}")
        scheduled.append(next_node)
        unscheduled.remove(next_node)

    new_graph = Graph()
    env = {}
    for node in scheduled:
        new_node = new_graph.node_copy(node, lambda n: env[n.name])
        env[node.name] = new_node

    return new_graph
