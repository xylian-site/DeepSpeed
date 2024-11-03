from ..profilers.graph_profile import MemoryProfilingInterpreter


def run_opt_passes(graph_id,
                   gm,
                   real_inputs,
                   opt_passes,
                   mem_prof,
                   profiling_results,
                   param_manager,
                   bwd,
                   debug_log=False):
    mem = profiling_results.bwd_mem if bwd else profiling_results.fwd_mem
    mem.clear()
    node_time = profiling_results.bwd_time if bwd else profiling_results.fwd_time
    tensor_sizes = profiling_results.bwd_tensor_sizes if bwd else profiling_results.fwd_tensor_sizes

    for i, opt_pass in enumerate(opt_passes):
        for name, current_alloc, delta in mem_prof.mem_record:
            mem.append((name, current_alloc, delta))

        opt_pass_fn, mem_budget = opt_pass

        graph = opt_pass_fn(gm.graph, graph_id, mem, node_time, tensor_sizes, mem_budget, param_manager, bwd)
        graph.lint()
        gm.graph = graph
        gm.recompile()

        if debug_log:
            print(f"Prefetching enabled for {'bwd' if bwd else 'fwd'} graph_id={graph_id} {graph}")

        mem_prof = MemoryProfilingInterpreter(gm)
        mem_prof.run(*real_inputs)
        if debug_log:
            mem_prof.dump(f"mem_prof_{'bwd' if bwd else 'fwd'}_{graph_id}_pass_{i}.csv")

    return gm
