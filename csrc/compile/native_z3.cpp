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

    long getId() const { return id_; }
    std::vector<int64_t> getShape() const { return shape_; }
    at::Tensor getDSTensor() const { return ds_tensor_; }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::Tensor ds_tensor_;
};

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void registerParam(long ds_id, const std::vector<int64_t>& ds_shape, at::Tensor ds_tensor)
    {
        params_.emplace(ds_id, DSParam(ds_id, ds_shape, ds_tensor));
    }

    DSParam getParam(long ds_id) { return params_.at(ds_id); }

private:
    std::unordered_map<long, DSParam> params_;
};

static DSParamRegistry registry = DSParamRegistry();
static c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;

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
    std::cout << "register_param ds_id=" << ds_id << " shape=" << ds_shape << std::endl;
    registry.registerParam(ds_id, ds_shape, ds_tensor);
}

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    std::cout << "set_process_group rank=" << pg->getRank() << std::endl;
    process_group = pg;
}

at::Tensor allgather(at::Tensor param_tensor, long ds_id)
{
    std::cout << "allgather called ds_id=" << ds_id << std::endl;
    return param_tensor;
}

TORCH_LIBRARY(native_z3, m)
{
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("test_call(Tensor a) -> Tensor");
    m.def("allgather(Tensor a, int id) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m)
{
    m.impl("test_call", &test_call);
    m.impl("allgather", &allgather);
}

TORCH_LIBRARY_IMPL(native_z3, CUDA, m)
{
    m.impl("test_call", &test_call);
    m.impl("allgather", &allgather);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_call", &test_call, "Test function");
    m.def("register_param", &register_param, "Register a parameter");
    m.def("set_process_group", &set_process_group, "Set the process group");
}
