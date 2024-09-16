// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "native_z3.h"

#define USE_C10D_NCCL

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
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
    std::unordered_map<std::string, size_t> op_n_args_;
    std::unordered_map<std::string, size_t> args_counter_;
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
    const size_t getNumParams() const { return params_.size(); }
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

class ReduceTask {
public:
    ReduceTask(long ds_id, at::Tensor send_buf) : ds_id_(ds_id), send_buf_(std::move(send_buf)) {}

    long getDSId() const { return ds_id_; }
    at::Tensor getSendBuf() const { return send_buf_; }

private:
    long ds_id_;
    at::Tensor send_buf_;
};

static DSParamRegistry param_registry = DSParamRegistry();
static c10::intrusive_ptr<c10d::ProcessGroup> process_group = nullptr;
static GraphOpStates op_states_fwd = GraphOpStates();
static GraphOpStates op_states_bwd = GraphOpStates();
static size_t reduce_counter;

static at::cuda::CUDAStream comm_stream = at::cuda::getStreamFromPool(false);
static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(false);
static ncclComm_t ncclComm;
static std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events;
static std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events;
static std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comp_done_events;
static std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comm_done_events;

static std::unordered_map<at::ScalarType, at::Tensor> reduce_in_buffers;
static std::unordered_map<at::ScalarType, at::Tensor> reduce_out_buffers;
static auto rs_done_event =
    std::shared_ptr<at::cuda::CUDAEvent>(new at::cuda::CUDAEvent(cudaEventDisableTiming));

static bool use_reduce_bucket = true;
static int64_t reduce_bucket_size = 5000000;
static std::unordered_map<at::ScalarType, size_t> reduce_bucket_offsets;
static std::unordered_map<at::ScalarType, at::Tensor> reduce_buckets;
static std::unordered_map<at::ScalarType, std::vector<ReduceTask>> reduce_tasks;

static bool profile = false;

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

void enable_profiling(bool enable) { profile = enable; }

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

void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent)
{
    // std::cout << "register_param ds_id=" << ds_id << " shape=" << ds_shape << std::endl;
    param_registry.registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent);
}

void register_op_n_args(const std::string& op_name, long n_args, bool is_backward)
{
    GraphOpStates& op_states = is_backward ? op_states_bwd : op_states_fwd;
    op_states.registerOpNArgs(op_name, n_args);
}

void init_comm(c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    process_group = pg;

    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);

    // ProcessGroup doesn't have an API to get the CUDA stream for comm calls.
    // So we create a NCCL communicator and call NCCL APIs directly.
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&ncclID),
                                    reinterpret_cast<uint8_t*>(&ncclID) + NCCL_UNIQUE_ID_BYTES);
    auto device = torch::Device(torch::kCUDA);
    at::Tensor tensor = torch::from_blob(vec.data(), {static_cast<long>(vec.size())}, torch::kUInt8)
                            .to(torch::Device(torch::kCUDA));
    std::vector<at::Tensor> bcast_input = {tensor};

    process_group->broadcast(bcast_input, c10d::BroadcastOptions())->wait();

    // create a new nccl communicator
    std::memcpy(&ncclID, tensor.to(torch::Device(torch::kCPU)).data_ptr(), NCCL_UNIQUE_ID_BYTES);
    ncclCommInitRank(&ncclComm, process_group->getSize(), ncclID, process_group->getRank());
}

void cleanup_comm()
{
    ncclCommDestroy(ncclComm);
    process_group = nullptr;
}

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
    reduce_counter = param_registry.getNumParams();
}

void end_backward(bool update)
{
    // unused
}

at::Tensor allgather_param(at::Tensor param_tensor, long ds_id)
{
    // std::cout << "allgather_param ds_id=" << ds_id << std::endl;

    const DSParam& param = param_registry.getParam(ds_id);

    if (param_registry.hasGatheredParam(ds_id)) { return param_registry.getGatheredParam(ds_id); }

    const at::Tensor& ds_tensor = param.getDSTensor();
    at::Tensor output_buf = torch::empty(param.getShape(), ds_tensor.options());

    const auto comp_done_event =
        std::shared_ptr<at::cuda::CUDAEvent>(new at::cuda::CUDAEvent(cudaEventDisableTiming));
    ag_comp_done_events[ds_id] = comp_done_event;
    comp_done_event->record();

    comp_done_event->block(comm_stream);
    ncclResult_t result = ncclAllGather(ds_tensor.contiguous().data_ptr(),
                                        output_buf.data_ptr(),
                                        ds_tensor.numel(),
                                        get_nccl_data_type(ds_tensor.scalar_type()),
                                        ncclComm,
                                        comm_stream);

    if (result != ncclSuccess) { throw std::runtime_error("NCCL AllGather failed"); }

    const auto comm_done_event =
        std::shared_ptr<at::cuda::CUDAEvent>(new at::cuda::CUDAEvent(cudaEventDisableTiming));
    ag_comm_done_events[ds_id] = comm_done_event;
    comm_done_event->record(comm_stream);

    if (profile) {
        // nccl calls run on a separate stream.
        // Events created in the profiler don't capture the time.

        comm_stream.synchronize();
    } else {
        param_registry.registerGatheredParam(ds_id, output_buf);
    }

    return output_buf;
}

at::Tensor allgather_param_meta(at::Tensor param_tensor, long ds_id)
{
    // std::cout << "allgather_param_meta ds_id=" << ds_id << std::endl;

    const DSParam& param = param_registry.getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options);
    return output_buf;
}

at::Tensor release_param(at::Tensor v, long ds_id)
{
    // std::cout << "release_param ds_id=" << ds_id << std::endl;

    if (profile and !param_registry.hasGatheredParam(ds_id)) {
        // Profiler runs this function multiple times.
        // We need to check if the gathered param is already released.
        return v;
    }

    const DSParam& param = param_registry.getParam(ds_id);
    if (!param.isPersistent()) {
        at::Tensor gathered_param = param_registry.getGatheredParam(ds_id);
        const auto options = gathered_param.options();
        at::Tensor empty_buffer = torch::empty({0}, options);
        gathered_param.set_data(empty_buffer);

        param_registry.unregisterGatheredParam(ds_id);
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
        ag_comm_done_events[ds_id]->block(at::cuda::getDefaultCUDAStream());
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

at::Tensor getOrExtendBuffer(std::unordered_map<at::ScalarType, at::Tensor>& buffers,
                             at::ScalarType scalar_type,
                             const at::Tensor& tensor)
{
    int64_t numel = tensor.numel();
    const auto options = at::TensorOptions().dtype(scalar_type).device(at::kCUDA);

    std::vector<int64_t> shape = {numel};
    if (!hasKey(buffers, scalar_type)) {
        buffers[scalar_type] = torch::empty(shape, options);
    } else {
        at::Tensor& buffer = buffers.at(scalar_type);
        if (buffer.numel() < numel) {
            buffer = torch::empty(shape, options);
            buffers[scalar_type] = buffer;
        }
    }
    return buffers.at(scalar_type);
}

at::Tensor getBufferFromBucket(at::ScalarType scalar_type, int64_t numel)
{
    if (!hasKey(reduce_buckets, scalar_type)) {
        const auto options = at::TensorOptions().dtype(scalar_type).device(at::kCUDA);
        reduce_buckets[scalar_type] = torch::empty({reduce_bucket_size}, options);
        reduce_bucket_offsets[scalar_type] = 0;
    }

    if (numel > reduce_bucket_size) {
        throw std::runtime_error("Buffer size exceeds the reduce bucket size");
    }

    size_t offset = reduce_bucket_offsets[scalar_type];
    if (offset + numel > reduce_bucket_size) {
        throw std::runtime_error("Buffer size exceeds the reduce bucket size");
    }

    reduce_bucket_offsets[scalar_type] += numel;
    return reduce_buckets[scalar_type].index({torch::indexing::Slice(offset, offset + numel)});
}

void flushReduceBucket(at::ScalarType scalar_type)
{
    if (!hasKey(reduce_tasks, scalar_type)) { return; }

    ncclGroupStart();
    for (const ReduceTask& t : reduce_tasks.at(scalar_type)) {
        auto recv_buf = param_registry.getParam(t.getDSId()).getGradBuffer();
        ncclResult_t result = ncclReduceScatter(t.getSendBuf().data_ptr(),
                                                recv_buf.data_ptr(),
                                                recv_buf.numel(),
                                                get_nccl_data_type(scalar_type),
                                                ncclAvg,
                                                ncclComm,
                                                nullptr);
        if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }
    }
    ncclGroupEnd();

    for (auto& it : reduce_bucket_offsets) { it.second = 0; }
    reduce_tasks.clear();
}

bool shouldFlushBucket(at::ScalarType scalar_type, int64_t numel)
{
    return reduce_bucket_offsets[scalar_type] > 0 &&
           reduce_bucket_offsets[scalar_type] + numel > reduce_bucket_size;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long ds_id)
{
    int world_size = process_group->getSize();
    const DSParam& param = param_registry.getParam(ds_id);
    const auto scalar_type = grad_tensor.scalar_type();

    if (shouldFlushBucket(scalar_type, grad_tensor.numel())) {
        flushReduceBucket(scalar_type);
    } else if (grad_tensor.numel() < reduce_bucket_size) {
        at::Tensor reduce_in_buffer =
            getBufferFromBucket(grad_tensor.scalar_type(), grad_tensor.numel());
        reduce_in_buffer.copy_(grad_tensor.contiguous().view({-1}));
        reduce_tasks[scalar_type].emplace_back(ds_id, grad_tensor);
    }

    reduce_counter--;

    if (reduce_counter == 0) { flushReduceBucket(scalar_type); }

    if (grad_tensor.numel() >= reduce_bucket_size) {
        auto recv_buf = param.getGradBuffer();
        ncclResult_t result = ncclReduceScatter(grad_tensor.data_ptr(),
                                                recv_buf.data_ptr(),
                                                recv_buf.numel(),
                                                get_nccl_data_type(scalar_type),
                                                ncclAvg,
                                                ncclComm,
                                                nullptr);
        if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }
    }

    if (profile) { at::cuda::device_synchronize(); }

    if (reduce_counter == 0) {
        // This looks duplicated but it's necessary to ensure the copy is done.
        // The backward hook has start_backward() might not be called when backward starting.
        reduce_counter = param_registry.getNumParams();
    }

    return at::Tensor();
}

// at::Tensor reduce_grad(at::Tensor grad_tensor, long ds_id)
// {
//     // lazy init of contiguous_grad_buf
//     if (reduce_bucket.numel() == 0) {
//         reduce_bucket = torch::empty({reduce_bucket_size}, grad_tensor.options());
//     }

//     int world_size = process_group->getSize();
//     const DSParam& param = param_registry.getParam(ds_id);

//     const auto comp_done_event =
//         std::shared_ptr<at::cuda::CUDAEvent>(new at::cuda::CUDAEvent(cudaEventDisableTiming));
//     rs_comp_done_events[ds_id] = comp_done_event;
//     auto comp_stream = at::cuda::getCurrentCUDAStream();

//     at::Tensor reduce_in_buffer =
//         getOrExtendBuffer(reduce_in_buffers, grad_tensor.scalar_type(), grad_tensor);
//     reduce_in_buffer = reduce_in_buffer.index(
//         {torch::indexing::Slice(0, grad_tensor.numel(), torch::indexing::None)});
//     at::Tensor reduce_out_buffer =
//         getOrExtendBuffer(reduce_out_buffers, grad_tensor.scalar_type(), param.getDSTensor());
//     reduce_out_buffer = reduce_out_buffer.index(
//         {torch::indexing::Slice(0, param.getDSTensor().numel(), torch::indexing::None)});
//     rs_done_event->block(comp_stream);

//     reduce_in_buffer.copy_(grad_tensor.contiguous().view({-1}));

//     comp_done_event->record(comp_stream);
//     comp_done_event->block(comm_stream);

//     ncclResult_t result = ncclReduceScatter(reduce_in_buffer.contiguous().data_ptr(),
//                                             reduce_out_buffer.data_ptr(),
//                                             reduce_out_buffer.numel(),
//                                             get_nccl_data_type(grad_tensor.scalar_type()),
//                                             ncclAvg,
//                                             ncclComm,
//                                             comm_stream);
//     if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }

//     {
//         c10::cuda::CUDAStreamGuard guard(comm_stream);
//         param.getGradBuffer().copy_(reduce_out_buffer, true);
//     }

//     rs_done_event->record(comm_stream);

//     if (profile) { at::cuda::device_synchronize(); }

//     reduce_counter--;

//     if (reduce_counter == 0) {
//         // This looks duplicated but it's necessary to ensure the copy is done.
//         // The backward hook has start_backward() might not be called when backward starting.
//         reduce_counter = param_registry.getNumParams();

//         // This synchronization ensures all of reduce calls are done before optimizer's step.
//         at::cuda::stream_synchronize(comm_stream);
//     }

//     return at::Tensor();
// }

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
    m.def("enable_profiling", &n3z::enable_profiling, "Enable profiling");
    m.def("init_comm", &n3z::init_comm, "Set the process group");
    m.def("cleanup_comm", &n3z::cleanup_comm, "Cleanup the process group");
    m.def("register_op_n_args",
          &n3z::register_op_n_args,
          "Register the number of arguments for an op");
    m.def("start_forward", &n3z::start_forward, "Start forward pass");
    m.def("end_forward", &n3z::end_forward, "End forward pass");
    m.def("start_backward", &n3z::start_backward, "Start backward pass");
    m.def("end_backward", &n3z::end_backward, "End backward pass");
}
