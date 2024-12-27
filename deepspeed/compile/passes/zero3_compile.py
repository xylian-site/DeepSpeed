import gc
from typing import List

from torch.fx import Node, GraphModule

from ..graph_param import DSGraphParamManager
from ..util import get_input_nodes, get_param_nodes, get_index_by_graph_id, get_deepcompile_handle
from ..fx import add_gather_and_release, add_gather_and_reduce, register_and_add_wait_allgather
from ..profilers.graph_profile import ProfilingInterpreter
from ..list_schedule import fast_free_schedule

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def add_z3_gather_release_fw(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results,
                            create_inputs_fn, param_manager,debug_log=False) -> GraphModule:

    nz3 = get_deepcompile_handle()
    graph = gm.graph

    real_inputs = create_inputs_fn()
    param_indices = profiling_results[graph_id].param_indices
    param_manager[graph_id] = DSGraphParamManager(graph, real_inputs, param_indices)

    graph = add_gather_and_release(graph_id, graph, param_manager[graph_id],
                                   get_param_nodes(graph, param_indices))

    nz3.register_graph(graph_id, [v[1] for v in param_indices])  # Need this before profiling

    profiler = ProfilingInterpreter(gm, debug_log=False)
    profiler.run(*real_inputs)
    del profiler
    gc.collect()
    get_accelerator().empty_cache()

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Fwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log)

    _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, False)
    nz3.register_graph_ops(graph_id, [n.name for n in ag_wait_nodes],
                            [len([arg for arg in n.args if isinstance(arg, Node)]) for n in ag_wait_nodes])

    return gm


def add_z3_gather_release_bw(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results,
                          create_inputs_fn, param_manager,debug_log=False) -> GraphModule:

    param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)
    gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw,
                                        param_name_to_grad)

    input_nodes = get_input_nodes(gm.graph)
    real_inputs = create_inputs_fn()
    assert len(input_nodes) == len(
        real_inputs), f"Expected {len(real_inputs)} inputs, got {len(input_nodes)}"

    nz3 = get_deepcompile_handle()
    real_outputs = ProfilingInterpreter(gm, debug_log=False).run(*real_inputs)

    del real_outputs
    gc.collect()
    get_accelerator().empty_cache()

    rank = dist.get_rank()
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    gm.graph = fast_free_schedule(gm.graph, get_accelerator().available_memory(), 0, debug_log=debug_log)

    _, ag_wait_nodes = register_and_add_wait_allgather(graph_id, gm.graph, True)
    nz3.register_bwd_graph_ops(graph_id, [n.name for n in ag_wait_nodes], [len(n.args) for n in ag_wait_nodes])

    return gm


def add_z3_gather_release(gm: GraphModule, graph_id: int, graph_order: List[int], profiling_results,
                          create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z3_gather_release_bw(gm, graph_id, graph_order, profiling_results, create_inputs_fn, param_manager)
    return add_z3_gather_release_fw(gm, graph_id, graph_order, profiling_results, create_inputs_fn, param_manager)
