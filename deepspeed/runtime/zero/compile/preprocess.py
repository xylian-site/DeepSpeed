from collections import defaultdict

import torch

from deepspeed.accelerator import get_accelerator
from .stage3_backend import profiling_results

WARMUP_STEPS: int = 5
MEM_MARGIN: int = 10_000_000_000


from .stage3_backend import param_manager, profiling_results


persistent_optimized = False


def sort_params_by_time_per_size():
    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)

    for graph_id, pm in param_manager.items():
        params = pm.params
        for param_name, param in params.items():
            ds_id = pm.ds_ids[param_name]
            ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

        profile = profiling_results[graph_id]
        for n in profile.fwd_graph.nodes:
            if n.target == torch.ops.native_z3.allgather_param:
                assert "tensor_size" in n.meta               
                ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                assert "device_time" in n.meta
                ds_id_to_time[n.args[2]] += n.meta["device_time"]

        if profile.bwd_graph is not None:
            for n in profile.bwd_graph.nodes:
                if n.target == torch.ops.native_z3.allgather_param:
                    assert "tensor_size" in n.meta
                    ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                    assert "device_time" in n.meta
                    ds_id_to_time[n.args[2]] += n.meta["device_time"]

    ds_ids = list(ds_id_to_size.keys())
    ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    # print(f"ds_id_to_size={ds_id_to_size}")
    # print(f"ds_id_to_time={ds_id_to_time}")
    for ds_id in ds_ids:
        print(f"ds_id={ds_id} time_per_size={ds_id_to_time[ds_id] / ds_id_to_size[ds_id]} time={ds_id_to_time[ds_id]} size={ds_id_to_size[ds_id]}")

    return ds_ids


def start_forward(nz3, micro_steps: int, global_steps: int, update: bool):
    accelerator = get_accelerator()

    global persistent_optimized
    if global_steps > WARMUP_STEPS and not persistent_optimized:
        max_alloc_mem = accelerator.max_memory_allocated()
        total_mem = accelerator.total_memory()
        available_mem = (total_mem - max_alloc_mem) - MEM_MARGIN

        print(f"global_steps={global_steps} Max memory allocated: {max_alloc_mem} Total memory: {total_mem} available_mem: {available_mem}")
        sort_params_by_time_per_size()
        persistent_optimized = True

    nz3.start_forward()
