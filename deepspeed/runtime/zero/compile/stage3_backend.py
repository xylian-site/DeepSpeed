# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict

import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from .fx import add_postprocess
from .schedule import schedule
from .graph_param import DSGraphParamManager

import os

pid = os.getpid()

gathered_params = {}
param_map = {}
z3_optimizer = None


def make_reduce_func(name: str):

    def reduce_grad(*x):
        if name in param_map:
            param = param_map[name]
            if param.ds_status != ZeroParamStatus.AVAILABLE:
                param.all_gather(param_list=[param])
            param.grad = x[0]
            z3_optimizer.reduce_ready_partitions_and_remove_grads(param)
            param.partition(param_list=[param], has_been_updated=False)
            return None
        if len(x) == 1:
            x = x[0]
        return x

    return reduce_grad


def add_allgather(graph: Graph, node: Node, ds_id):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.allgather_param,
                           extra_args=[ds_id],
                           name=f"allgather_ds_param_{node.target}")


def add_release(graph: Graph, node: Node, release_node: Node, ds_id):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.release_param,
                           extra_args=[ds_id],
                           name=f"release_ds_param_{release_node.target}")


def add_reduce(graph: Graph, grad_node: Node, param_name: str):
    return add_postprocess(graph, grad_node, make_reduce_func(param_name), name=f"reduce_ds_param_{param_name}")


def add_gather_and_release(gm: GraphModule, param_nodes: List[Node], ds_ids: Dict[str, int]):
    graph = gm.graph
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    user_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        user_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)

    allgather_nodes = {}
    for pn in param_nodes:
        allgather_nodes[pn] = add_allgather(graph, pn, ds_ids[pn.name])

    release_nodes = {}
    for v, nodes in user_nodes.items():
        for node in nodes:
            assert v not in release_nodes
            release_nodes[v] = add_release(graph, node, v, ds_ids[v.name])

    return allgather_nodes, release_nodes


def add_gather_and_reduce(gm: GraphModule, param_manager: DSGraphParamManager):
    for pn in param_manager.param_nodes_bw:
        n = add_allgather(gm.graph, pn, ds_id=param_manager.ds_ids[pn.name])
        param_manager.add_allgather_node(pn.name, n, bw=True)

    for pn in param_manager.param_nodes:
        rn = add_reduce(gm.graph, param_manager.get_grad_name(pn.name), pn.name)
        param_manager.add_release_node(pn.name, rn, bw=True)


graph_counts = defaultdict(int)
param_manager = None


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}.dot"

        g = FxGraphDrawer(graph, fname)
        with open(f"{name}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


def make_stage3_backend(dump_graphs=False):

    def stage3_backend(gm: GraphModule, sample_inputs):
        # n_params = len(list(gm.named_parameters()))
        # for name, param in gm.named_parameters():
        #     print(f"stage3_backend 1 param: {name} {param.shape} {param.ds_id}")
        param_ds_ids = [param.ds_id for _, param in gm.named_parameters()]

        def fw(gm, sample_inputs):
            global param_manager
            param_manager = DSGraphParamManager(gm.graph, sample_inputs, param_ds_ids)

            # for param in sample_inputs:
            #     print(f"stage3_backend 2 input: {name} {param.shape}")

            dump_graph(gm, f"forward_aot", skip=not dump_graphs)

            allgather_nodes, release_nodes = add_gather_and_release(gm, param_manager.param_nodes,
                                                                    param_manager.ds_ids)
            for pn, an in allgather_nodes.items():
                param_manager.add_allgather_node(pn.name, an)
            for pn, rn in release_nodes.items():
                param_manager.add_release_node(pn.name, rn)

            dump_graph(gm, f"forward_aot_comm", skip=not dump_graphs)
            gm.graph = schedule(gm.graph, param_manager)
            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            param_manager.add_bw_graph(gm.graph)

            dump_graph(gm, f"backward_aot", skip=not dump_graphs)

            add_gather_and_reduce(gm, param_manager)

            dump_graph(gm, f"backward_aot_comm", skip=not dump_graphs)
            # gm.graph = schedule(gm.graph, param_manager, bw=True)
            # dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
