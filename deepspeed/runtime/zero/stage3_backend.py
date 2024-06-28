from copy import deepcopy
import argparse

import torch
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from typing import Callable, Any
import torch
from torch.fx import Node, Graph, symbolic_trace, GraphModule, Proxy
import operator


def make_gather_func(name: str):

    def gather_fn():
        torch.library.define(f"ds_compile::gather_param_{name}", "(Tensor x) -> Tensor")

        def f(x):
            if hasattr(x, "all_gather"):
                x.all_gather(param_list=[x])
            return x

        torch.library.impl(f"ds_compile::gather_param_{name}", ["cpu", "cuda"], f)
        return f
    
    return gather_fn()


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


def wrap_func_node(graph):
    new_graph = Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                Proxy(env[x.name], tracer) if isinstance(x, Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node


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

            for n in gm.graph.nodes:
                print(f"1 node: {n} {n.op} {n.target} {n.kwargs} {n.users}")

            # param_nodes = get_param_nodes(gm.graph, n_params)
            # for pn in param_nodes:
            #     print(f"param: {pn}")
            param_users = get_param_users(gm.graph, n_params)
            print(f"param_users: {param_users}")

            # print(f"2 GraphModule: {gm.__class__} graph: {graph.__class__}")
            # for n, p in gm.named_parameters():
            #     print(f"2 param: {n} {p.shape} {p.device}")

            trans = Z3GraphTransformer(gm.graph)
            # trans.insert_fn_after(lambda n: n.op == "placeholder", torch.ops.ds_compile.gather_param)
            trans.insert_gather_with_name_after()
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
