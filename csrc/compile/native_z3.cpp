// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace n3z {

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

class GraphOpStates {
public:
    GraphOpStates() {}
    ~GraphOpStates() {}

    void registerOpNArgs(const std::string& op_name, long n_args)
    {
        op_n_args_[op_name] = n_args;
        args_counter_[op_name] = n_args;
    }

    void resetArgCounter()
    {
        // std::cout << "resetArgCounter size op_n_args_ " << op_n_args_.size() << std::endl;

        for (const auto& it : op_n_args_) {
            assert(hasKey(op_n_args_, it.first));
            args_counter_[it.first] = op_n_args_.at(it.first);
        }
    }

    void decrementArgCounter(const std::string& op_name)
    {
        // std::cout << "decrementArgCounter " << op_name << std::endl;

        assert(hasKey(args_counter_, op_name));
        if (args_counter_.at(op_name) == 0) return;
        args_counter_[op_name]--;
    }

    long getArgCounter(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name);
    }

    bool isArgCounterZero(const std::string& op_name) const
    {
        assert(hasKey(args_counter_, op_name));
        return args_counter_.at(op_name) == 0;
    }

private:
    std::unordered_map<std::string, long> op_n_args_;
    std::unordered_map<std::string, long> args_counter_;
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
    bool hasGatheredParam(long ds_id) const { return hasKey(gathered_params_, ds_id); }

    void addAllgatherHandle(long ds_id, c10::intrusive_ptr<c10d::Work> handle)
    {
        allgather_handles_[ds_id] = handle;
    }

    c10::intrusive_ptr<c10d::Work> getAllgatherHandle(long ds_id) const
    {
        assert(hasKey(allgather_handles_, ds_id));
        return allgather_handles_.at(ds_id);
    }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
    std::unordered_map<long, c10::intrusive_ptr<c10d::Work>> allgather_handles_;
};

static DSParamRegistry registry = DSParamRegistry();
static c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;
static GraphOpStates op_states_fwd = GraphOpStates();
static GraphOpStates op_states_bwd = GraphOpStates();
static at::cuda::CUDAStream reduce_stream = at::cuda::getStreamFromPool(true);

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

void register_op_n_args(const std::string& op_name, long n_args, bool is_backward)
{
    GraphOpStates& op_states = is_backward ? op_states_bwd : op_states_fwd;
    op_states.registerOpNArgs(op_name, n_args);
}

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg) { process_group = pg; }

void start_forward()
{
    // std::cout << "start_forward" << std::endl;
    op_states_fwd.resetArgCounter();
}

void end_forward()
{
    // std::cout << "end_forward" << std::endl;
}

void start_backward(bool update)
{
    // std::cout << "start_backward update=" << update << std::endl;
    op_states_bwd.resetArgCounter();
}

void end_backward(bool update)
{
    // unused
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
    registry.addAllgatherHandle(ds_id, handle);

    registry.registerGatheredParam(ds_id, output_buf);

    return output_buf;
}

at::Tensor allgather_param_meta(at::Tensor param_tensor, long ds_id)
{
    const DSParam& param = registry.getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options);
    return output_buf;
}

at::Tensor release_param(at::Tensor v, long ds_id)
{
    const DSParam& param = registry.getParam(ds_id);
    if (!param.isPersistent()) {
        at::Tensor gathered_param = registry.getGatheredParam(ds_id);
        const auto options = gathered_param.options();
        at::Tensor empty_buffer = torch::empty({0}, options);
        gathered_param.set_data(empty_buffer);

        registry.unregisterGatheredParam(ds_id);
    }

    return v;
}

at::Tensor release_param_meta(at::Tensor v, long ds_id) { return v; }

at::Tensor wait_allgather(at::Tensor v,
                          long ds_id,
                          const std::string& user,
                          long n_args,
                          bool is_backward)
{
    GraphOpStates& op_states = is_backward ? op_states_bwd : op_states_fwd;

    op_states.decrementArgCounter(user);

    if (op_states.isArgCounterZero(user)) {
        auto handle = registry.getAllgatherHandle(ds_id);
        handle->wait();
    }

    return v;
}

at::Tensor wait_allgather_meta(at::Tensor v,
                               long ds_id,
                               const std::string& user,
                               long n_args,
                               bool is_backward)
{
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
    // {
    //     c10::cuda::CUDAStreamGuard guard(reduce_stream);
    //     handle->wait();
    //     grad_buf /= world_size;
    //     param.getGradBuffer().copy_(grad_buf);
    // }
    handle->wait();
    grad_buf /= world_size;
    param.getGradBuffer().copy_(grad_buf);

    return at::Tensor();
}

at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long ds_id) { return at::Tensor(); }

}  // namespace n3z

TORCH_LIBRARY(native_z3, m)
{
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("test_call(Tensor a) -> Tensor");
    m.def("allgather_param(Tensor a, int id) -> Tensor");
    m.def("release_param(Tensor a, int id) -> Tensor");
    m.def("wait_allgather(Tensor a, int id, str user, int n_args, bool bwd) -> Tensor");
    m.def("reduce_grad(Tensor a, int id) -> Tensor");
}

TORCH_LIBRARY_IMPL(native_z3, CPU, m)
{
    m.impl("test_call", &n3z::test_call);
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);
}

TORCH_LIBRARY_IMPL(native_z3, CUDA, m)
{
    m.impl("test_call", &n3z::test_call);
    m.impl("allgather_param", &n3z::allgather_param);
    m.impl("release_param", &n3z::release_param);
    m.impl("wait_allgather", &n3z::wait_allgather);
    m.impl("reduce_grad", &n3z::reduce_grad);
}

TORCH_LIBRARY_IMPL(native_z3, Meta, m)
{
    m.impl("allgather_param", &n3z::allgather_param_meta);
    m.impl("release_param", &n3z::release_param_meta);
    m.impl("wait_allgather", &n3z::wait_allgather_meta);
    m.impl("reduce_grad", &n3z::reduce_grad_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test_call", &n3z::test_call, "Test function");
    m.def("register_param", &n3z::register_param, "Register a parameter");
    m.def("set_process_group", &n3z::set_process_group, "Set the process group");
    m.def("register_op_n_args",
          &n3z::register_op_n_args,
          "Register the number of arguments for an op");
    m.def("start_forward", &n3z::start_forward, "Start forward pass");
    m.def("end_forward", &n3z::end_forward, "End forward pass");
    m.def("start_backward", &n3z::start_backward, "Start backward pass");
    m.def("end_backward", &n3z::end_backward, "End backward pass");
}
