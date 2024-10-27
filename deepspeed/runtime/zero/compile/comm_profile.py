# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

from scipy.interpolate import interp1d

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def sync_all():
    get_accelerator().synchronize()
    dist.barrier()


def get_bw(comm_op, size, duration):
    n = dist.get_world_size()
    tput = 0
    busbw = 0

    if duration == 0:
        raise ValueError("Error. Duration is 0.")

    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        raise ValueError("wrong comm_op specified")

    return tput, busbw


# Run all_gather and print metrics
def timed_all_gather(input, output, start_event, end_event, warmup, trials, async_op):
    sync_all()
    # Warmups, establish connections, etc.
    for i in range(warmup):
        dist.all_gather_into_tensor(output, input, async_op=async_op)
    sync_all()

    # time the actual comm op trials times and average it
    start_event.record()
    for i in range(trials):
        dist.all_gather_into_tensor(output, input, async_op=async_op)
    end_event.record()
    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000

    # maintain and clean performance data
    avg_duration = duration / trials
    size = input.element_size() * input.nelement()
    # tput, busbw = get_bw('all_gather', size, avg_duration)

    return size, avg_duration


def run_all_gather(device, dtype, maxsize, warmup=5, trials=10, async_op=False):

    # Prepare benchmark header
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    start_event = get_accelerator().Event(enable_timing=True)
    end_event = get_accelerator().Event(enable_timing=True)

    # Create list of message sizes
    M_LIST = []
    for x in (2**p for p in range(1, maxsize)):
        M_LIST.append(x)

    results = []
    sync_all()
    # loop over various tensor sizes
    for M in M_LIST:
        global_rank = dist.get_rank()
        try:
            mat = torch.ones(world_size, M, dtype=dtype, device=device)
            sync_all()
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            get_accelerator().empty_cache()
            output = torch.zeros(input.nelement() * world_size, dtype=dtype, device=device)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of GPU memory. Exiting comm op.')
                sync_all()
                break
            else:
                raise e
        sync_all()
        results.append(timed_all_gather(input, output, start_event, end_event, warmup, trials, async_op))

    return results


def create_predictor(results):
    # Extract size and avg_duration from results
    sizes = [result[0] for result in results]
    durations = [result[1] for result in results]

    # Create a linear interpolation function
    return interp1d(sizes, durations, kind='cubic', fill_value="extrapolate")


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)
    print(f"local_rank={local_rank}")

    deepspeed.init_distributed(dist_backend='nccl')

    device = get_accelerator().current_device()
    results = run_all_gather(device, torch.bfloat16, 30)

    if dist.get_rank() == 0:
        for size, avg_duration in results:
            print(f"size: {size}, avg_duration: {avg_duration}")

        # Create predictor function
        predictor = create_predictor(results)

        # Predict time for a specific data size
        example_size = 1e9
        predicted_time = predictor(example_size)
        print(f"Predicted time for size {example_size}: {predicted_time:.6f} seconds")

    dist.destroy_process_group()
