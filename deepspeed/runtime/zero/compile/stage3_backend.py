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

import deepspeed.comm as dist
from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes

from .fx import add_postprocess, add_args_process
# from .schedule import schedule
from .graph_param import DSGraphParamManager
from .profile import ProfilingInterpreter
from .list_schedule import list_schedule2
from .util import get_param_nodes

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
                           name=f"allgather_ds_param_{node.target}_{ds_id}",
                           meta={
                               "param_name": node.name,
                               "ds_id": ds_id,
                               "tensor_meta": node.meta["tensor_meta"],
                               "comm": True
                           })


def add_release(graph: Graph, node: Node, release_node: Node, ds_id: int):
    return add_postprocess(graph,
                           node,
                           torch.ops.native_z3.release_param,
                           extra_args=[ds_id],
                           name=f"release_ds_param_{release_node.target}_{ds_id}",
                           meta={
                               "param_name": node.name,
                               "ds_id": ds_id,
                               "tensor_meta": node.meta["tensor_meta"],
                               "comm": False
                           })


def add_wait_allgather(graph: Graph, node: Node, ds_id: int, user: str, n_args: int, bwd: bool):
    return add_args_process(graph,
                            node,
                            torch.ops.native_z3.wait_allgather,
                            extra_args=[ds_id, user, n_args, bwd],
                            name=f"wait_allgather_ds_param_{ds_id}",
                            meta={
                                "param_name": node.name,
                                "ds_id": ds_id,
                                "tensor_meta": node.meta["tensor_meta"],
                                "comm": False
                            })


def add_reduce(graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    return add_postprocess(graph,
                           grad_node,
                           torch.ops.native_z3.reduce_grad,
                           extra_args=[ds_id],
                           name=f"reduce_ds_param_{param_name}",
                           meta={
                               "grad_name": grad_node.name,
                               "ds_id": ds_id,
                               "tensor_meta": grad_node.meta["tensor_meta"],
                               "comm": True
                           })


def _add_wait_allgather(graph: Graph, bwd: bool):

    for node in graph.nodes:
        ag_args = [
            arg for arg in node.args if isinstance(arg, Node) and arg.target == torch.ops.native_z3.allgather_param
        ]
        if len(ag_args) > 0:
            assert len(ag_args) == 1, f"Node {node.name} takes multiple allgathered params"
            nz3.register_op_n_args(node.name, len(node.args), bwd)
            ds_id = ag_args[0].meta["ds_id"]
            add_wait_allgather(graph, node, ds_id, node.name, len(node.args), bwd)


def add_gather_and_release(graph: Graph, param_manager: DSGraphParamManager, param_nodes: List[Node]):
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    last_user_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        last_user_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)

    for pn in param_nodes:
        add_allgather(graph, pn, param_manager.ds_ids[pn.name])

    for pn, nodes in last_user_nodes.items():
        for node in nodes:
            add_release(graph, node, pn, param_manager.ds_ids[pn.name])


def add_gather_and_reduce(graph: Graph, param_manager: DSGraphParamManager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]):

    add_gather_and_release(graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        add_reduce(graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])


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

            add_gather_and_release(gm.graph, param_manager, get_param_nodes(gm.graph, len(param_ds_ids)))
            ProfilingInterpreter(gm, nz3).run(*sample_inputs)

            dump_graph(gm, f"forward_aot_comm", skip=not dump_graphs)
            gm.graph = list_schedule2(gm.graph)

            _add_wait_allgather(gm.graph, False)
            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            gm.recompile()
            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            param_nodes_bw, param_name_to_grad = param_manager.get_bwd_mapping(gm.graph)

            dump_graph(gm, f"backward_aot", skip=not dump_graphs)

            add_gather_and_reduce(gm.graph, param_manager, param_nodes_bw, param_name_to_grad)
            ProfilingInterpreter(gm, nz3).run(*sample_inputs)

            dump_graph(gm, f"backward_aot_comm", skip=not dump_graphs)
            gm.graph = list_schedule2(gm.graph)
            _add_wait_allgather(gm.graph, True)
            dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)
            gm.recompile()
            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
