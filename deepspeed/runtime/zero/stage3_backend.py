from copy import deepcopy
import argparse

import torch
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.compile.tracer import ParamLifetimeCheckTracer, add_dependency_on_params
from deepspeed.runtime.zero.compile.nx import fx_to_nx, find_reachable_terminal_nodes

from typing import Callable, Any
import torch
from torch.fx import Node, Graph, symbolic_trace, GraphModule, Proxy
import operator


import os
pid = os.getpid()

def make_allgather_func(name: str):

    # torch.library.define(f"ds_compile::gather_param_{name}", "(Tensor x) -> Tensor")

    def allgather_param(x):
        print(f"[{pid}] allgather_fn {name}")
        if hasattr(x, "all_gather"):
            x.all_gather(param_list=[x])
        return x

    # torch.library.impl(f"ds_compile::gather_param_{name}", ["cpu", "cuda"], allgather_param)
    return allgather_param


def make_preprocess_func(name: str):

    def preprocess_fn():
        def f(x):
            print(f"[{pid}] preprocess_fn {name}")
            return x
        return f
    
    return preprocess_fn()


def make_postprocess_func(name: str):

    def postprocess_fn():
        def f(x):
            print(f"[{pid}] postprocess_fn {name}")
            return x
        return f
    
    return postprocess_fn()


class Z3GraphTransformer:
    def __init__(self, g: Graph) -> None:
        self.g = g

    def insert_fn_before(self,
                         fn: Callable[..., Any],
                         node_cond_fn: Callable[[Node], bool],
                         in_cond_fn: Callable[[Node], bool]=lambda _: True) -> None:
        for n in self.g.nodes:
            if node_cond_fn(n):
                for in_node in n.all_input_nodes:
                    if in_cond_fn(in_node):
                        with self.g.inserting_before(n):
                            new_node = self.g.call_function(fn, (in_node,), {})
                            n.replace_input_with(in_node, new_node)

    def insert_fn_after(self, cond_fn: Callable[[Node], bool], fn: Callable[..., Any]) -> None:
        print(f"insert_fn_after cond_fn={cond_fn}, fn={fn}")
        users = {}
        for n in self.g.nodes:
            print(f"insert_fn_after n={n} class={n.__class__}")
            if cond_fn(n):
                with self.g.inserting_after(n):
                    node_users = n.users.keys()
                    print(n)
                    new_node = self.g.call_function(fn, (n,), {})
                    for u in node_users:
                        if u != new_node:
                            users[u] = (n, new_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)

    def insert_gather_with_name_after(self) -> None:
        cond_fn = lambda n: n.op == "placeholder"
        # print(f"insert_gather_with_name_after cond_fn={cond_fn}")
        users = {}
        for n in self.g.nodes:
            # print(f"insert_gather_with_name_after n={n} class={n.__class__}")
            if cond_fn(n):
                with self.g.inserting_after(n):
                    node_users = n.users.keys()
                    new_node = self.g.call_function(make_gather_func(n), (n,), {})
                    for u in node_users:
                        if u != new_node:
                            users[u] = (n, new_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)



def get_param_nodes(graph: Graph, n_params: int):
    return [n for n in graph.nodes if n.op == "placeholder"][:n_params]


def get_param_users(graph: Graph, n_params: int):
    param_nodes = get_param_nodes(graph, n_params)
    return {p: set(p.users.keys()) for p in param_nodes}


def add_postprocess(graph: Graph, node: Node, fn: Callable[..., Any]):
    # https://github.com/pytorch/examples/blob/main/fx/wrap_output_dynamically.py
    with graph.inserting_after(node):
        node_users = node.users.keys()
        new_node = graph.call_function(fn(node.target), (node,), {})
        users = {}
        for u in node_users:
            if u != new_node:
                users[u] = (node, new_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)


def add_allgather(graph: Graph, node: Node):
    add_postprocess(graph, node, make_allgather_func)


backend_count = 0
fw_count = 0
def make_stage3_backend(module: torch.nn.Module):
    def stage3_backend(gm: GraphModule, sample_inputs): 
        global backend_count

        graph = gm.graph
        n_params = len(list(gm.named_parameters()))

        graph.process_inputs(sample_inputs)

        # # Forward compiler capture
        def fw(gm, sample_inputs):
            global fw_count
            g = FxGraphDrawer(gm, 'fn')
            with open(f"forward_aot_{backend_count}_{fw_count}.svg", "wb") as file:
                file.write(g.get_dot_graph().create_svg())
            
            param_nodes = get_param_nodes(gm.graph, n_params)
            new_graph = add_dependency_on_params(gm.graph, param_nodes)
            for n in new_graph.nodes:
                print(f"node: {n} {n.op} {n.target} {n.kwargs} users={n.users} required_inputs={n.required_inputs}")

            nx_graph = fx_to_nx(new_graph)
            release_nodes = {}
            for pn in param_nodes:
                dependent_nodes = [n for n in new_graph.nodes if pn in n.required_inputs]
                release_nodes[pn] = find_reachable_terminal_nodes(nx_graph, dependent_nodes)
            print(f"release_nodes: {release_nodes}")
                
            param_users = get_param_users(gm.graph, n_params)
            for pn in param_nodes:
                add_allgather(gm.graph, pn)

            # for v, node in release_nodes.items():
            #     add_postprocess(gm.graph, node, make_postprocess_func)

            trans = Z3GraphTransformer(gm.graph)
            # trans.insert_fn_after(lambda n: n.op == "placeholder", torch.ops.ds_compile.gather_param)
            # trans.insert_gather_with_name_after()
            gm.recompile()

            g = FxGraphDrawer(gm, 'fn')
            with open(f"forward_aot_{backend_count}_{fw_count}_mod.svg", "wb") as file:
                file.write(g.get_dot_graph().create_svg())

            fw_count += 1

            return make_boxed_func(gm.forward)
        
        # # # Call AOTAutograd
        aot_mod = aot_module_simplified(gm, sample_inputs,
                                        fw_compiler=fw)

        backend_count += 1

        return aot_mod

    return stage3_backend
