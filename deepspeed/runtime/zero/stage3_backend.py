from typing import Dict
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

from deepspeed.runtime.zero.compile.tracer import add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes, sort_nodes_by_distance_to_output, serialize
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from typing import Callable, Any, List, Set
import torch
from torch.fx import Node, Graph, GraphModule

from .compile.fx import get_output_node, add_postprocess
from .compile.schedule import schedule

import os
pid = os.getpid()

gathered_params = {}
param_map = {}
z3_optimizer = None


def make_allgather_func(name: str):

    # torch.library.define(f"ds_compile::gather_param_{name}", "(Tensor x) -> Tensor")

    def allgather_param(x):
        # print(f"allgather_param {name} {x.__class__} ds_id={hasattr(x, 'ds_id')} ds_param={hasattr(x, 'ds_param')} all_gather={hasattr(x, 'all_gather')}")
        if hasattr(x, 'ds_id'):
            x.all_gather(param_list=[x])
            global gathered_params
            gathered_params[name] = x

            global param_map
            param_map[name] = x
        return x

    # torch.library.impl(f"ds_compile::gather_param_{name}", ["cpu", "cuda"], allgather_param)
    return allgather_param


def make_release_func(name: str):

    def release_param(*x):
        global gathered_params
        if name in gathered_params:
            param = gathered_params.pop(name)
            param.partition(param_list=[param], has_been_updated=False)
        if len(x) == 1:
            x = x[0]
        return x

    return release_param


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


def add_allgather(graph: Graph, node: Node):
    add_postprocess(graph, node, make_allgather_func(node.target))


def add_release(graph: Graph, node: Node, release_node: Node):
    add_postprocess(graph, node, make_release_func(release_node.target))


def add_reduce(graph: Graph, grad_node: Node, param_name: str):
    add_postprocess(graph, grad_node, make_reduce_func(param_name))

def add_gather_and_release(gm: GraphModule, param_nodes: List[Node]):
    graph = gm.graph
    add_dependency_on_params(graph, param_nodes)

    nx_graph = fx_to_nx(graph)
    release_nodes = {}
    for pn in param_nodes:
        dependent_nodes = [n for n in graph.nodes if pn in n.required_inputs]
        release_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)
        
    for pn in param_nodes:
        add_allgather(graph, pn)

    for v, nodes in release_nodes.items():
        for node in nodes:
            add_release(graph, node, v)

    gm.recompile()


@dataclass
class DSGraphParam:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    node: Node
    allgather_node: Node
    release_node: Node
    param: torch.Tensor


class DSGraphParamManager:
    def __init__(self, fw_graph: Graph, sample_inputs: Any, n_params: int):
        self._fw_graph = fw_graph
        self._params: Dict[str, DSGraphParam] = {}
        self._param_name_to_grad: Dict[str, Node] = {}

        self._param_nodes = [n for n in self._fw_graph.nodes if n.op == "placeholder"][:n_params]
        param_inputs = sample_inputs[:n_params]
        
        for pn, pi in zip(self.param_nodes, param_inputs):
            self._params[pn.name] = DSGraphParam(
                name=pn.name,
                shape=pi.size(),
                dtype=pi.dtype,
                device=pi.device,
                node=pn,
                allgather_node=None,
                release_node=None,
                param=pi
            )

    def add_bw_graph(self, bw_graph: Graph):
        self._bw_graph = bw_graph

        output_node = get_output_node(self._bw_graph)
        self._param_nodes_bw = [n for n in self._bw_graph.nodes if n.name in self.param_names]
        self._param_name_to_grad = {param_node.name: grad for param_node, grad in zip(self.param_nodes, output_node.args[0])}

    @property
    def param_nodes(self):
        return self._param_nodes

    @property
    def param_names(self):
        return [pn.name for pn in self.param_nodes]

    @property
    def param_nodes_bw(self):
        return self._param_nodes_bw

    def get_grad_name(self, param_name):
        assert self._param_name_to_grad is not None, "Backward graph is not added yet"
        return self._param_name_to_grad[param_name]


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
        n_params = len(list(gm.named_parameters()))

        def fw(gm, sample_inputs):
            global param_manager
            param_manager = DSGraphParamManager(gm.graph, sample_inputs, n_params)

            dump_graph(gm, f"forward_aot", skip=not dump_graphs)

            add_gather_and_release(gm, param_manager.param_nodes)
            schedule(gm.graph)

            dump_graph(gm, f"forward_aot_scheduled", skip=not dump_graphs)

            return make_boxed_func(gm.forward)

        def bw(gm, sample_inputs):
            param_manager.add_bw_graph(gm.graph)

            dump_graph(gm, f"backward_aot", skip=not dump_graphs)

            for pn in param_manager.param_nodes_bw:
                add_allgather(gm.graph, pn)
            for pn in param_manager.param_nodes:
                add_reduce(gm.graph, param_manager.get_grad_name(pn.name), pn.name)

            gm.recompile()

            dump_graph(gm, f"backward_aot_scheduled", skip=not dump_graphs)

            return make_boxed_func(gm.forward)

        # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs,
                                        fw_compiler=fw,
                                        bw_compiler=bw)
        aot_mod = torch._dynamo.optimize()(aot_mod)

        return aot_mod

    return stage3_backend
