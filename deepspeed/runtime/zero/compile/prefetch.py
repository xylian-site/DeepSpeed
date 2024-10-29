from typing import List, Tuple

import torch
from torch.fx import Graph, Node

from deepspeed.accelerator import get_accelerator

from .comm_profile import create_predictor


FUSE_FACTOR = 0.8
MARGIN = 5_000_000_000


run_prefetch_pass = False


def enable_prefetch():
    global run_prefetch_pass
    run_prefetch_pass = True


def is_prefetch_enabled():
    return run_prefetch_pass


def schedule_prefetch(graph: Graph, graph_id: int, mem: List[Tuple[str, int, int]], op_time: List[Tuple[str, int, int]], tensor_sizes: List[Tuple[str, int]]):
    max_mem = get_accelerator().available_memory() - get_accelerator().memory_allocated() - MARGIN

    print(f"schedule_prefetch max_mem={max_mem} available_memory={get_accelerator().available_memory()} memory_allocated={get_accelerator().memory_allocated()} margin={MARGIN}")

    mem_dict = {name: (alloc_mem, delta) for name, alloc_mem, delta in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    comm_predictor = create_predictor()

    order_rev = list(reversed(graph.nodes))
    new_order_rev = []
    prefetch_ags = []
    prefetch_ag_groups = []
    ag_tensor_size_sum = 0
    for i, node in enumerate(order_rev):
        print(f"Checking node reverse order {node.name} {node.target} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}")
    
        if node.op != "placeholder":
            assert i < len(order_rev) - 1
            assert node.name in mem_dict
            next_node = order_rev[i + 1]
            next_alloc_mem, _ = mem_dict[next_node.name]

            # Free up memory
            while next_alloc_mem + ag_tensor_size_sum > max_mem:
                if len(prefetch_ag_groups) > 0:
                    # launch prefetch
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in fused_ag_nodes])
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)                    
                    print(f"Free up memory fused_ag_nodes={fused_ag_nodes} total_ag_tensor_size={total_ag_tensor_size} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}")
                    continue
                elif len(prefetch_ags) > 0:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                    ag_tensor_size_sum = 0
                    print(f"Free up memory prefetch_ags={prefetch_ag_groups} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}")
                else:
                    break

            if node.target == torch.ops.native_z3.allgather_param:

                pred_time_current = comm_predictor(ag_tensor_size_sum)
                pred_time_next = comm_predictor(tensor_size_dict[node.name])
                pred_time_fused = comm_predictor(ag_tensor_size_sum + tensor_size_dict[node.name])

                print(f"found allgather_param ag_tensor_size_sum={ag_tensor_size_sum} tensor_size_dict[node.name]={tensor_size_dict[node.name]} pred_time_current={pred_time_current} pred_time_next={pred_time_next} pred_time_fused={pred_time_fused} (pred_time_current + pred_time_next)={pred_time_current + pred_time_next}")

                if len(prefetch_ags) > 0 and pred_time_fused > FUSE_FACTOR * (pred_time_current + pred_time_next):
                    # stop fusing here
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                    print(f"stop fusing prefetch_ags={prefetch_ag_groups}")
                else:
                    print(f"continue fusing ag_tensor_size_sum={ag_tensor_size_sum} ag_size={tensor_size_dict[node.name]} prefetch_ags={prefetch_ags}")
                prefetch_ags.append(node)
                ag_tensor_size_sum += tensor_size_dict[node.name]

        # print(f"node={node} next_alloc_mem={next_alloc_mem} pending_ags={len(prefetch_ags)} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}")

        new_order_rev.append(node)

        if node.op != "placeholder" and order_rev[i + 1].op == "placeholder":
            # last flush
            all_remaining_ags = prefetch_ags + [ag_node for ag_group in prefetch_ag_groups for ag_node in ag_group]
            if len(all_remaining_ags) > 0:
                new_order_rev.append(all_remaining_ags)
            print(f"flush last prefetch_ags ds_ids={all_remaining_ags}")

    new_graph = Graph()
    env = {}
    for node in reversed(new_order_rev):
        if isinstance(node, Node):
            print(f"reconstruct {node.name} {node.target}")
            new_node = new_graph.node_copy(node, lambda n: env[n.name])
            env[node.name] = new_node
        else:
            param_nodes = [ag_node.args[0] for ag_node in node]
            param_nodes_copy = [env[param_node.name] for param_node in param_nodes]

            ds_ids = [ag_node.args[2] for ag_node in node]
            new_graph.call_function(torch.ops.native_z3.prefetch_params_fused, args=(graph_id, param_nodes_copy, ds_ids))
            print(f"Found prefetch group {node}")
    
    return new_graph