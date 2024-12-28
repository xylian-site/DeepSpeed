// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#define USE_C10D_NCCL

namespace dc {

c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;
c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem = nullptr;
ncclComm_t nccl_comm;
bool use_symm_mem;
bool profile = false;
bool pre_div_reduce = true;

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size)
{
    c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());
    std::vector<int64_t> sizes = {size};
    std::vector<int64_t> strides = {1};
    at::Tensor sym_mem_ws = c10d::symmetric_memory::empty_strided_p2p(
        {size}, {1}, c10::ScalarType::Byte, device, process_group->getGroupName(), std::nullopt);
    return c10d::symmetric_memory::rendezvous(sym_mem_ws);
}

ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclHalf;
        case at::kDouble: return ncclDouble;
        case at::kBFloat16: return ncclBfloat16;
        case at::kLong: return ncclInt64;
        case at::kInt: return ncclInt;
        case at::kChar: return ncclInt8;
        default: throw std::runtime_error("Unsupported scalar type");
    }
}

}  // namespace dc
