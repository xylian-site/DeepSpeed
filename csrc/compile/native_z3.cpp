// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

class DSParam {
public:
    DSParam(long id,
            std::vector<int64_t> ds_shape,
            at::Tensor ds_tensor,
            at::Tensor grad_buffer,
            bool persistent)
        : id_(id),
          shape_(std::move(ds_shape)),
          ds_tensor_(ds_tensor),
          grad_buffer_(grad_buffer),
          persistent_(persistent)
    {
    }

    long getId() const { return id_; }
    std::vector<int64_t> getShape() const { return shape_; }
    at::Tensor getDSTensor() const { return ds_tensor_; }
    at::Tensor getGradBuffer() const { return grad_buffer_; }
    bool isPersistent() const { return persistent_; }

private:
    long id_;
    std::vector<int64_t> shape_;
    at::Tensor ds_tensor_;
    at::Tensor grad_buffer_;
    bool persistent_;
};

class DSParamRegistry {
public:
    DSParamRegistry() {}
    ~DSParamRegistry() {}

    void registerParam(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent)
    {
        params_.emplace(ds_id, DSParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent));
    }

    void registerGatheredParam(long ds_id, at::Tensor ds_tensor)
    {
        gathered_params_.emplace(ds_id, ds_tensor);
    }

    void unregisterGatheredParam(long ds_id) { gathered_params_.erase(ds_id); }

    const DSParam& getParam(long ds_id) const { return params_.at(ds_id); }
    const at::Tensor& getGatheredParam(long ds_id) const { return gathered_params_.at(ds_id); }
    bool hasGatheredParam(long ds_id) const { return gathered_params_.count(ds_id) > 0; }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
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

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent)
{
    // std::cout << "register_param ds_id=" << ds_id << " shape=" << ds_shape << std::endl;
    registry.registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent);
}

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    std::cout << "set_process_group rank=" << pg->getRank() << std::endl;
    process_group = pg;
}

at::Tensor allgather_param(at::Tensor param_tensor, long ds_id)
{
    // std::cout << "allgather_param ds_id=" << ds_id << std::endl;

    const DSParam& param = registry.getParam(ds_id);

    if (registry.hasGatheredParam(ds_id)) { return registry.getGatheredParam(ds_id); }

    at::Tensor output_buf = torch::empty(param.getShape(), param.getDSTensor().options());
    std::vector<at::Tensor> outputs = {output_buf};
    std::vector<at::Tensor> inputs = {param.getDSTensor()};
    c10::intrusive_ptr<c10d::Work> handle =
        process_group->allgather_into_tensor_coalesced(outputs, inputs);
    handle->wait();  // necessary

    registry.registerGatheredParam(ds_id, output_buf);

    return output_buf;
}

at::Tensor release_param(at::Tensor v, long ds_id)
{
    const DSParam& param = registry.getParam(ds_id);
    if (!param.isPersistent()) {
        // This just removes the reference to the tensor
        registry.unregisterGatheredParam(ds_id);
    }

    return v;
}

at::Tensor wait_allgather(at::Tensor v, long ds_id, long n_args)
{
    // std::cout << "wait_allgather ds_id=" << ds_id << " n_args=" << n_args << std::endl;

    return v;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long ds_id)
{
    int world_size = process_group->getSize();
    const DSParam& param = registry.getParam(ds_id);

    at::Tensor grad_buf = torch::empty_like(param.getDSTensor());
    std::vector<at::Tensor> outputs = {grad_buf};
    std::vector<at::Tensor> inputs = {grad_tensor};
    c10::intrusive_ptr<c10d::Work> handle =
        process_group->reduce_scatter_tensor_coalesced(outputs, inputs);
    handle->wait();
    grad_buf /= world_size;

    param.getGradBuffer().copy_(grad_buf);

    if (!param.isPersistent()) { registry.unregisterGatheredParam(ds_id); }

    return at::Tensor();
}

TORCH_LIBRARY(native_z3, m)
{
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("test_call(Tensor a) -> Tensor");
    m.def("allgather_param(Tensor a, int id) -> Tensor");
    m.def("release_param(Tensor a, int id) -> Tensor");
    m.def("wait_allgather(Tensor a, int id, int n_args) -> Tensor");
    m.def("reduce_grad(Tensor a, int id) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m)
{
    m.impl("test_call", &test_call);
    m.impl("allgather_param", &allgather_param);
    m.impl("release_param", &release_param);
    m.impl("wait_allgather", &wait_allgather);
    m.impl("reduce_grad", &reduce_grad);
}

TORCH_LIBRARY_IMPL(native_z3, CUDA, m)
{
    m.impl("test_call", &test_call);
    m.impl("allgather_param", &allgather_param);
    m.impl("release_param", &release_param);
    m.impl("wait_allgather", &wait_allgather);
    m.impl("reduce_grad", &reduce_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_call", &test_call, "Test function");
    m.def("register_param", &register_param, "Register a parameter");
    m.def("set_process_group", &set_process_group, "Set the process group");
}
