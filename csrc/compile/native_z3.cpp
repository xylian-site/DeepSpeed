// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

class DSParam {
public:
    DSParam(long id, std::vector<int64_t> ds_shape, at::Tensor ds_tensor)
        : id_(id), shape_(std::move(ds_shape)), ds_tensor_(ds_tensor)
    {
    }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::Tensor ds_tensor_;
};

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void register_param(long ds_id, const std::vector<int64_t>& ds_shape, at::Tensor ds_tensor)
    {
        params_.emplace_back(ds_id, ds_shape, ds_tensor);
    }

private:
    std::vector<DSParam> params_;
};

static DSParamRegistry registry = DSParamRegistry();
static bool is_dist_initialized = false;

at::Tensor test_call(at::Tensor param)
{
    std::cout << "test_call " << param << std::endl;
    return param;
}

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes)
{
    std::vector<int64_t> result;
    for (int i = 0; i < sizes.size(); i++) { result.push_back(sizes[i]); }
    return result;
}

void register_param(long ds_id, const std::vector<int64_t>& ds_shape, at::Tensor ds_tensor)
{
    std::cout << "register_param " << ds_id << " shape " << ds_shape << std::endl;
    registry.register_param(ds_id, ds_shape, ds_tensor);
}

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    std::cout << "set_process_group rank=" << pg->getRank() << std::endl;
}

TORCH_LIBRARY(native_z3, m)
{
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m) { m.impl("test_call", &test_call); }

TORCH_LIBRARY_IMPL(native_z3, CUDA, m) { m.impl("test_call", &test_call); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_call", &test_call, "Test function");
    m.def("register_param", &register_param, "Register a parameter");
    m.def("set_process_group", &set_process_group, "Set the process group");
}
