from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import defaultdict
import torch
from torch.fx import Tracer, Node, Graph
from torch.fx.proxy import Proxy, GraphAppendingTracer
import torch.utils._pytree as pytree


aten = torch._ops.ops.aten
ops_reuse_inputs = [
    aten.t.default
]

def retrace(graph: Graph) -> Graph:
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    new_graph = Graph()
    env = {}
    reuse_inputs = defaultdict(list)

    tracer = GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function':
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                Proxy(env[x.name], tracer) if isinstance(x, Node) else x for x in node.args]
            output_proxy = node.target(*proxy_args)

            print(f"retracing target={node.target} args={node.args} output_proxy={output_proxy} output_proxy.node={output_proxy.node}")

            # for a in node.args:
            #     for param, users in reuse_inputs.items():
            #         if a == param:
            #             for user in users:
            #                 reuse_inputs[a].append(user)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node

            if new_node.target in ops_reuse_inputs:
                for a in new_node.args:
                    for param, users in reuse_inputs.items():
                        if a in users:
                            users.append(new_node)

        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node

            if node.op == 'placeholder':
                reuse_inputs[node].append(new_node)

        new_node.required_inputs = []
        for a in new_node.args:
            for param, users in reuse_inputs.items():
                if a in users:
                    new_node.required_inputs.append(param)

    return new_graph


def fake_to_real(a):
    def to_real_tensor(t):
        return t

    return pytree.tree_map_only(torch.Tensor, to_real_tensor, a)



class ParamLifetimeCheckTracer(Tracer):
        
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    def __init__(self):
        super().__init__()
        self.reuse_inputs = defaultdict(list)

    def create_node(self, kind : str, target : Union[str, Callable],
                    args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)

        if n.target in self.ops_reuse_inputs:
            for a in args:
                if isinstance(a, Node):
                    self.reuse_inputs[a].append(n)
        # n.reuse_inputs = 
        # print(f"create_node kind={kind} target={target} args={args} kwargs={kwargs} name={name} type_expr={type_expr}")
        return n

    def check_lifetime(self, node):
        if node in self.reuse_inputs:
            for user in self.reuse_inputs[node]:
                self.check_lifetime(user)
        print(f"check_lifetime node={node}")


# Let's use this custom tracer to trace through this module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(x) + torch.ones(3, 4)

# mod = MyModule()

# traced_graph = ParamLifetimeCheckTracer().trace(mod)
# # trace() returns a Graph. Let's wrap it up in a
# # GraphModule to make it runnable
# traced = torch.fx.GraphModule(mod, traced_graph)
