# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict

import torch
from torch.fx import Node, Graph, GraphModule
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes

from .fx import add_postprocess
# from .schedule import schedule
from .graph_param import DSGraphParamManager
from .profile import ProfilingInterpreter

import os

pid = os.getpid()

gathered_params = {}
param_map = {}
z3_optimizer = None
nz3 = None


def add_allgather(graph: Graph, node: Node, ds_id: int):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.allgather_param,
                           extra_args=[ds_id],
                           name=f"allgather_ds_param_{node.target}_{ds_id}")


def add_release(graph: Graph, node: Node, release_node: Node, ds_id: int):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.release_param,
                           extra_args=[ds_id],
                           name=f"release_ds_param_{release_node.target}_{ds_id}")


def add_wait_allgather(graph: Graph, node: Node, ds_id: int, user: str, n_args: int, bwd: bool):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.wait_allgather,
                           extra_args=[ds_id, user, n_args, bwd],
                           name=f"wait_allgather_ds_param_{ds_id}")


def add_reduce(graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    return add_postprocess(graph,
                           grad_node,
                           torch.ops.native_z3.reduce_grad,
                           extra_args=[ds_id],
                           name=f"reduce_ds_param_{param_name}")


def _add_wait_allgather(graph: Graph, param_manager: DSGraphParamManager, bwd: bool):
    ds_ids = param_manager.ds_ids

    def allgathered_param_args(node):
        return [
            arg for arg in node.args if isinstance(arg, Node) and arg.target == torch.ops.native_z3.allgather_param
        ]

    for node in graph.nodes:
        ag_args = allgathered_param_args(node)
        if len(ag_args) > 0:
            nz3.register_op_n_args(node.name, len(ag_args), bwd)
            for arg in ag_args:
                param_name = param_manager.allgather_param_name(arg, bw=bwd)
                add_wait_allgather(graph, arg, ds_ids[param_name], node.name, len(ag_args), bwd)


def add_gather_and_release(gm: GraphModule, param_manager: DSGraphParamManager):
    graph = gm.graph
    param_nodes = param_manager.param_nodes
    ds_ids = param_manager.ds_ids

    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    last_user_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        last_user_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)

    allgather_nodes = {}
    for pn in param_nodes:
        allgather_nodes[pn] = add_allgather(graph, pn, ds_ids[pn.name])

    release_nodes = {}
    for pn, nodes in last_user_nodes.items():
        for node in nodes:
            assert pn not in release_nodes
            release_nodes[pn] = add_release(graph, node, pn, ds_ids[pn.name])

    for pn, an in allgather_nodes.items():
        param_manager.add_allgather_node(pn.name, an)
    for pn, rn in release_nodes.items():
        param_manager.add_release_node(pn.name, rn)


def add_gather_and_reduce(gm: GraphModule, param_manager: DSGraphParamManager):
    graph = gm.graph
    param_nodes_bw = param_manager.param_nodes_bw
    ds_ids = param_manager.ds_ids

    add_dependency_on_params(graph, param_nodes_bw)

    for pn in param_nodes_bw:
        n = add_allgather(gm.graph, pn, param_manager.ds_ids[pn.name])
        param_manager.add_allgather_node(pn.name, n, bw=True)

    for pn in param_manager.param_nodes:
        rn = add_reduce(gm.graph, param_manager.get_grad_name(pn.name), pn.name, param_manager.ds_ids[pn.name])
        param_manager.add_release_node(pn.name, rn, bw=True)


graph_counts = defaultdict(int)
param_manager = None


def dump_graph(graph: GraphModule, name: str, skip=False):
    if not skip and dist.get_rank() == 0:
        global graph_counts
        fname = f"{name}_{graph_counts[name]}"
        graph.graph.print_tabular()

        g = FxGraphDrawer(graph, fname)
        with open(f"{fname}.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())

        graph_counts[name] += 1


def make_stage3_backend(dump_graphs=False):
    from deepspeed.ops.op_builder import NativeZ3Builder
    global nz3
    nz3 = NativeZ3Builder().load()

    def stage3_backend(gm: GraphModule, sample_inputs):
        param_ds_ids = [param.ds_id for _, param in gm.named_parameters()]

        def fw(gm, sample_inputs):
            global param_manager
            param_manager = DSGraphParamManager(gm.graph, sample_inputs, param_ds_ids)

            dump_graph(gm, f"forward_aot", skip=not dump_graphs)

            add_gather_and_release(gm, param_manager)
            ProfilingInterpreter(gm).run(*sample_inputs)

            dump_graph(gm, f"forward_aot_comm", skip=not dump_graphs)
            # gm.graph = schedule(gm.graph, param_manager)
            _add_wait_allgather(gm.graph, param_manager, False)
            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            param_manager.add_bw_graph(gm.graph)

            dump_graph(gm, f"backward_aot", skip=not dump_graphs)

            add_gather_and_reduce(gm, param_manager)

            dump_graph(gm, f"backward_aot_comm", skip=not dump_graphs)
            # gm.graph = schedule(gm.graph, param_manager, bw=True)
            _add_wait_allgather(gm.graph, param_manager, True)
            dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
