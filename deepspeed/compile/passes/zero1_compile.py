# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
from typing import List

from torch.fx import Node, GraphModule

from ..graph_param import DSGraphParamManager
from ..profilers.graph_profile import ProfilingInterpreter
from ..list_schedule import fast_free_schedule

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

NAME = "zero3_compile"


def add_z1_reduce_bw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[int],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:

    param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)

    print(f"add_z1_reduce_bw param_nodes_bw={param_nodes_bw} param_name_to_grad={param_name_to_grad}")

    # gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

    # input_nodes = get_input_nodes(gm.graph)
    # real_inputs = create_inputs_fn()
    # assert len(input_nodes) == len(real_inputs), f"Expected {len(real_inputs)} inputs, got {len(input_nodes)}"

    # nz3 = get_deepcompile_handle()
    # real_outputs = ProfilingInterpreter(gm, debug_log=False).run(*real_inputs)

    # del real_outputs
    # gc.collect()
    # get_accelerator().empty_cache()

    # rank = dist.get_rank()
    # graph_index = get_index_by_graph_id(graph_order, graph_id)
    # if rank == 0 and debug_log:
    #     print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    # gm.graph = fast_free_schedule(gm.graph, get_accelerator().available_memory(), 0, debug_log=debug_log)

    # _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)
    # nz3.register_bwd_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

    return gm


def add_z1_reduce(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results, create_inputs_fn,
                          mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z1_reduce_bw(gm, graph_id, graph_order, profiling_results, create_inputs_fn, param_manager)
    return gm
