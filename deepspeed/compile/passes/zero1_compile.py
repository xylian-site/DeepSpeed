# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import torch
from torch.fx import GraphModule

from ..fx import add_postprocess, move_primals_to_head, _make_node_meta

NAME = "zero1_compile"


def add_z1_reduce_bw(gm: GraphModule, graph_id: int, param_manager) -> GraphModule:

    graph = gm.graph
    pm = param_manager[graph_id]
    param_nodes_bw, param_name_to_grad = pm.get_bwd_mapping(graph)

    for param_name in pm.param_names:

        def debug_reduce_op(x: torch.Tensor, graph_id: int, param_name: str):
            print(f"add_reduce debug_reduce_op x={x.shape} graph_id={graph_id} param_name={param_name}")
            return x

        grad_node = param_name_to_grad[param_name]
        add_postprocess(graph,
                        grad_node,
                        debug_reduce_op,
                        extra_args=[graph_id, param_name],
                        name=f"reduce_param_{param_name}",
                        meta=_make_node_meta(grad_node, param_name, True))

    gm.graph = move_primals_to_head(graph)
    return gm


def add_z1_reduce(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results, create_inputs_fn,
                  mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z1_reduce_bw(gm, graph_id, param_manager)
    return gm
