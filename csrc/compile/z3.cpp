// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#define USE_C10D_NCCL

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace dc {

class Z3CustomOpExecutor : public CustomOpExecutor {
public:
    Z3CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                       std::shared_ptr<DSParamRegistry> param_registry,
                       std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                       std::vector<long> ds_ids,
                       ncclComm_t nccl_comm,
                       at::cuda::CUDAStream ag_stream,
                       at::cuda::CUDAStream rs_stream,
                       at::cuda::CUDAStream copy_stream,
                       at::cuda::CUDAStream offload_stream,
                       at::cuda::CUDAStream reload_stream,
                       bool pre_div_reduce)
        : CustomOpExecutor(process_group,
                           param_registry,
                           reduce_buckets,
                           ds_ids,
                           nccl_comm,
                           rs_stream,
                           copy_stream,
                           pre_div_reduce),
          ag_stream_(ag_stream),
          offload_stream_(offload_stream),
          reload_stream_(reload_stream)
    {
        for (long ds_id : ds_ids_) {
            ag_comm_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            ag_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }
        reduce_counter_ = ds_ids_.size();
    }
    ~Z3CustomOpExecutor() {}

    void endBackward() override
    {
        if (param_updated_) {
            for (auto& it : has_acc_grad_) {
                it.second = false;
                param_registry_->setValid(it.first, false);
            }
        }

        for (auto& it : reload_buffers_) {
            it.second.record_stream(at::cuda::getCurrentCUDAStream());
        }
        reload_buffers_.clear();
    }

    void launchAllGather(at::Tensor output_buf,
                         long ds_id,
                         c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        const DSParam& param = param_registry_->getParam(ds_id);
        const at::Tensor& ds_tensor = param.getDSTensor();

        if (symm_mem == nullptr) {
            ncclResult_t result = ncclAllGather(ds_tensor.contiguous().data_ptr(),
                                                output_buf.data_ptr(),
                                                ds_tensor.numel(),
                                                get_nccl_data_type(ds_tensor.scalar_type()),
                                                nccl_comm_,
                                                ag_stream_);

            if (result != ncclSuccess) { throw std::runtime_error("NCCL AllGather failed"); }
        } else {
            at::cuda::CUDAStreamGuard guard(ag_stream_);
            int world_size = process_group_->getSize();
            int rank = process_group_->getRank();

            at::Tensor local_buf =
                symm_mem->get_buffer(rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
            local_buf.copy_(ds_tensor, true);

            symm_mem->barrier(0);
            auto chunks = output_buf.flatten().chunk(world_size);
            for (int step = 0; step < world_size; step++) {
                int remote_rank = (rank - step + world_size) % world_size;
                auto src_buf = symm_mem->get_buffer(
                    remote_rank, ds_tensor.sizes(), ds_tensor.scalar_type(), 0);
                chunks[remote_rank].copy_(src_buf.flatten(), true);
            }
            symm_mem->barrier(0);
        }

        param_registry_->registerGatheredParam(ds_id, output_buf);
        param_registry_->setValid(ds_id, true);
    }

    at::Tensor allgatherParam(long ds_id,
                              c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        if (param_registry_->isValid(ds_id)) { return param_registry_->getGatheredParam(ds_id); }

        const DSParam& param = param_registry_->getParam(ds_id);
        const at::Tensor& ds_tensor = param.getDSTensor();
        at::Tensor output_buf = param_registry_->hasGatheredParam(ds_id)
                                    ? param_registry_->getGatheredParam(ds_id)
                                    : torch::empty(param.getShape(), ds_tensor.options());

        assert(hasKey(ag_comp_done_events_, ds_id));
        ag_comp_done_events_[ds_id]->record();
        ag_comp_done_events_[ds_id]->block(ag_stream_);

        launchAllGather(output_buf, ds_id, symm_mem);

        ag_comm_done_events_[ds_id]->record(ag_stream_);
        return output_buf;
    }

    void prefetchParamsFused(std::vector<int64_t> ds_ids,
                             c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem)
    {
        std::vector<int64_t> invalid_ds_ids;
        for (const auto& ds_id : ds_ids) {
            if (!param_registry_->isValid(ds_id)) { invalid_ds_ids.push_back(ds_id); }
        }

        std::unordered_map<long, at::Tensor> output_bufs;
        for (long ds_id : invalid_ds_ids) {
            const DSParam& param = param_registry_->getParam(ds_id);
            if (param_registry_->hasGatheredParam(ds_id)) {
                output_bufs[ds_id] = param_registry_->getGatheredParam(ds_id);
            } else {
                output_bufs[ds_id] = torch::empty(param.getShape(), param.getDSTensor().options());
            }
        }

        for (long ds_id : invalid_ds_ids) {
            ag_comp_done_events_[ds_id]->record();
            ag_comp_done_events_[ds_id]->block(ag_stream_);
        }

        ncclGroupStart();
        for (long ds_id : invalid_ds_ids) {
            assert(hasKey(output_bufs, ds_id));
            launchAllGather(output_bufs.at(ds_id), ds_id, symm_mem);
        }
        ncclGroupEnd();

        for (long ds_id : invalid_ds_ids) { ag_comm_done_events_[ds_id]->record(ag_stream_); }
    }

    void releaseParam(long ds_id)
    {
        const DSParam& param = param_registry_->getParam(ds_id);

        if (!param.isPersistent()) {
            at::Tensor gathered_param = param_registry_->getGatheredParam(ds_id);

            if (gathered_param.defined()) {  // gathered param is undefined while profiling
                const auto options = gathered_param.options();
                at::Tensor empty_buffer = torch::empty({0}, options);
                gathered_param.set_data(empty_buffer);
            }

            param_registry_->unregisterGatheredParam(ds_id);
        }
    }

    at::Tensor waitAllgather(at::Tensor v,
                             const std::vector<long>& ds_ids,
                             const std::string& user,
                             long n_args,
                             bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;

        op_states.decrementArgCounter(user);

        if (op_states.isArgCounterZero(user)) {
            for (long ds_id : ds_ids) {
                assert(hasKey(ag_comm_done_events_, ds_id));
                ag_comm_done_events_[ds_id]->block(at::cuda::getCurrentCUDAStream());
            }
            op_states_fwd_.resetArgCounter();
        }

        return v;
    }

    at::Tensor offloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(offload_events_, id)) {
            offload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            offload_comp_done_events_[id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            const auto options = at::TensorOptions().pinned_memory(true).device(torch::kCPU);
            offload_buffers_[id] = at::empty_like(tensor, options);
        }

        offload_comp_done_events_[id]->record();
        offload_comp_done_events_[id]->block(offload_stream_);
        {
            at::cuda::CUDAStreamGuard guard(offload_stream_);
            offload_buffers_.at(id).copy_(tensor, true);
        }

        tensor.record_stream(offload_stream_);

        offload_events_[id]->record(offload_stream_);
        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor reloadTensor(at::Tensor tensor, long id)
    {
        if (!hasKey(reload_events_, id)) {
            reload_events_[id] = std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }

        assert(hasKey(offload_buffers_, id));
        offload_events_[id]->block(reload_stream_);

        at::Tensor ten;
        {
            at::cuda::CUDAStreamGuard guard(reload_stream_);

            assert(hasKey(offload_buffers_, id));
            at::Tensor buf = offload_buffers_.at(id);
            const auto options = at::TensorOptions().device(torch::kCUDA);
            ten = at::empty_like(buf, options);
            ten.copy_(buf, true);

            reload_buffers_[id] = ten;
        }

        reload_events_[id]->record(reload_stream_);
        return ten;
    }

    at::Tensor waitOffload(at::Tensor tensor, long id)
    {
        assert(hasKey(offload_events_, id));
        offload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(offload_buffers_, id));
        return offload_buffers_.at(id);
    }

    at::Tensor waitReload(at::Tensor tensor, long id)
    {
        assert(hasKey(reload_events_, id));
        reload_events_[id]->block(at::cuda::getCurrentCUDAStream());

        assert(hasKey(reload_buffers_, id));
        auto ten = reload_buffers_.at(id);

        // We can't release here because the tensor is still being used
        // We will need "freeReloadedTensor" after the last user of the tensor to call
        // ".record_stream". As it is a bit complicated, we clear the buffer and do at the end of
        // the backward pass for now. reload_buffers_.erase(id);
        return ten;
    }

    bool hasReloadBuffer(long id) { return hasKey(reload_buffers_, id); }

    bool hasParam(long ds_id) const { return hasKey(has_acc_grad_, ds_id); }

private:
    at::cuda::CUDAStream ag_stream_;
    at::cuda::CUDAStream offload_stream_;
    at::cuda::CUDAStream reload_stream_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events_;

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> offload_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> reload_events_;
    std::unordered_map<long, at::Tensor> offload_buffers_;
    std::unordered_map<long, at::Tensor> reload_buffers_;
};

static std::shared_ptr<DSParamRegistry> param_registry;
static std::unordered_map<long, std::shared_ptr<Z3CustomOpExecutor>> executors;
std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets = nullptr;

static at::cuda::CUDAStream ag_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream rs_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream copy_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream offload_stream = at::cuda::getStreamFromPool(true);
static at::cuda::CUDAStream reload_stream = at::cuda::getStreamFromPool(true);

std::vector<int64_t> sizes_to_int_vector(at::IntArrayRef sizes)
{
    std::vector<int64_t> result;
    for (int i = 0; i < sizes.size(); i++) { result.push_back(sizes[i]); }
    return result;
}

void lazy_init_symm_memory()
{
    if (use_symm_mem && !symm_mem) {
        int64_t max_param_size = 0;
        for (const auto& it : param_registry->getParams()) {
            int64_t size = it.second.getDSTensor().numel() * it.second.getDSTensor().element_size();
            if (size > max_param_size) { max_param_size = size; }
        }
        symm_mem = getSymmMemWorkspace(max_param_size);
    }
}

void enable_profiling(bool enable) { profile = enable; }

bool is_profiling() { return profile; }

void register_graph(long graph_id, const std::vector<long>& ds_ids)
{
    executors[graph_id] = std::make_shared<Z3CustomOpExecutor>(process_group,
                                                               param_registry,
                                                               reduce_buckets,
                                                               ds_ids,
                                                               nccl_comm,
                                                               ag_stream,
                                                               rs_stream,
                                                               copy_stream,
                                                               offload_stream,
                                                               reload_stream,
                                                               pre_div_reduce);
}

void register_graph_ops(long graph_id,
                        const std::vector<std::string>& op_names,
                        const std::vector<long>& n_args)
{
    assert(op_names.size() == n_args.size());
    for (int i = 0; i < op_names.size(); i++) {
        executors[graph_id]->registerOpNArgs(op_names[i], n_args[i], false);
    }
}

void register_bwd_graph_ops(long graph_id,
                            const std::vector<std::string>& op_names,
                            const std::vector<long>& n_args)
{
    assert(hasKey(executors, graph_id));
    for (int i = 0; i < op_names.size(); i++) {
        executors[graph_id]->registerOpNArgs(op_names[i], n_args[i], true);
    }
}

void init_z3(c10::intrusive_ptr<c10d::ProcessGroup> pg,
             int64_t initial_reduce_bucket_size,
             bool enable_double_buffer,
             bool _use_symm_mem)
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
    ncclCommInitRank(&nccl_comm, process_group->getSize(), ncclID, process_group->getRank());

    param_registry = std::make_shared<DSParamRegistry>();
    reduce_buckets = std::make_shared<DoubleBufferedReduceBucket>(initial_reduce_bucket_size,
                                                                  enable_double_buffer);
    use_symm_mem = _use_symm_mem;
}

void reset_z3()
{
    executors.clear();
    // We keep the buckets for memory estimation
    // reduce_buckets->clear();
}

void cleanup_z3()
{
    reset_z3();
    cleanup();
}

void register_z3_param(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent)
{
    param_registry->registerParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent);
    if (persistent) { param_registry->registerGatheredParam(ds_id, ds_tensor); }
}

at::Tensor allgather_param(at::Tensor param_tensor, long graph_id, long ds_id)
{
    assert(hasKey(executors, graph_id));
    return executors[graph_id]->allgatherParam(ds_id, symm_mem);
}

void set_persistent(long ds_id)
{
    param_registry->setPersistent(ds_id, true);

    // Allocate buffer here
    // Memory fragmentation will be more severe if we allocate in forward/backward
    for (auto& it : executors) {
        if (it.second->hasParam(ds_id)) { it.second->allgatherParam(ds_id, symm_mem); }
    }
}

void prefetch_params_fused(long graph_id,
                           const std::vector<at::Tensor> params,
                           const std::vector<long>& ds_ids)
{
    executors[graph_id]->prefetchParamsFused(ds_ids, symm_mem);
}

// for profiling
void invalidate_gathered_param(long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    if (param.isPersistent()) { return; }

    param_registry->unregisterGatheredParam(ds_id);
    param_registry->registerGatheredParam(ds_id, at::Tensor());
}

void clear_all_gathered_params()
{
    for (const auto& it : param_registry->getParams()) {
        long ds_id = it.first;
        const DSParam& param = param_registry->getParam(ds_id);
        if (param.isPersistent()) { continue; }
        if (param_registry->hasGatheredParam(ds_id)) {
            param_registry->unregisterGatheredParam(ds_id);
        }
    }
}

at::Tensor allgather_param_meta(at::Tensor param_tensor, long graph_id, long ds_id)
{
    const DSParam& param = param_registry->getParam(ds_id);
    auto options = param.getDSTensor().options().device(c10::kMeta);
    at::Tensor output_buf = torch::empty(param.getShape(), options);
    return output_buf;
}

void release_param(long graph_id, long ds_id) { executors[graph_id]->releaseParam(ds_id); }

at::Tensor wait_allgather(at::Tensor v,
                          long graph_id,
                          const std::vector<long>& ds_ids,
                          const std::string& user,
                          long n_args,
                          bool is_backward)
{
    executors[graph_id]->waitAllgather(v, ds_ids, user, n_args, is_backward);
    return v;
}

at::Tensor wait_allgather_meta(at::Tensor v,
                               long graph_id,
                               const std::vector<long>& ds_ids,
                               const std::string& user,
                               long n_args,
                               bool is_backward)
{
    return v;
}

at::Tensor reduce_grad(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    if (!profile) { executors[graph_id]->reduceGrad(grad_tensor, ds_id); }
    return at::Tensor();
}

void free_tensors(std::vector<at::Tensor> tensors)
{
    if (!profile) {
        for (auto& tensor : tensors) {
            if (tensor.is_cuda()) {
                tensor.record_stream(at::cuda::getCurrentCUDAStream());
                tensor.set_data(torch::empty({0}, tensor.options()));
            }
        }
    }
}

at::Tensor reduce_grad_meta(at::Tensor grad_tensor, long graph_id, long ds_id)
{
    return at::Tensor();
}

at::Tensor offload_tensor(at::Tensor tensor, long graph_id, long id)
{
    // auto dims = tensor.sizes();
    // std::cout << "offload_tensor graph_id=" << graph_id << " id=" << id
    //     << " dim=" << join_as_str(dims, ",") << std::endl;
    return executors[graph_id]->offloadTensor(tensor, id);
}

at::Tensor reload_tensor(at::Tensor tensor, long graph_id, long id)
{
    // auto dims = tensor.sizes();
    // std::cout << "reload_tensor graph_id=" << graph_id << " id=" << id
    //     << " dim=" << join_as_str(dims, ",") << std::endl;
    return executors[graph_id]->reloadTensor(tensor, id);
}

at::Tensor wait_offload(at::Tensor tensor, long graph_id, long id)
{
    return executors[graph_id]->waitOffload(tensor, id);
}

at::Tensor wait_reload(at::Tensor tensor, long graph_id, long id)
{
    if (profile && !executors[graph_id]->hasReloadBuffer(id)) { return tensor; }

    return executors[graph_id]->waitReload(tensor, id);
}

void start_forward()
{
    lazy_init_symm_memory();
    for (auto& it : executors) { it.second->startForward(); }
}

void end_forward()
{
    for (auto& it : executors) { it.second->endForward(); }
}

void start_backward(bool update)
{
    for (auto& it : executors) { it.second->startBackward(update); }
}

// We don't call this
// void end_backward(bool update)
// {
// }

at::Tensor test_call(at::Tensor a)
{
    std::cout << "test_call" << std::endl;
    return a;
}

}  // namespace dc

TORCH_LIBRARY(dc, m)
{
    m.def("allgather_param(Tensor a, int graph_id, int id) -> Tensor");
    m.def("prefetch_params_fused(int graph_id, Tensor[] params, int[] ids) -> ()");
    m.def(
        "wait_allgather(Tensor a, int graph_id, int[] ids, str user, int n_args, bool bwd) -> "
        "Tensor");
    m.def("reduce_grad(Tensor a, int graph_id, int id) -> Tensor");
    m.def("free_tensors(Tensor[] a) -> ()");
    m.def("offload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("reload_tensor(Tensor a, int id, int id) -> Tensor");
    m.def("wait_offload(Tensor a, int id, int id) -> Tensor");
    m.def("wait_reload(Tensor a, int id, int id) -> Tensor");

    m.def("test_call(Tensor a) -> Tensor");
}

TORCH_LIBRARY_IMPL(dc, CPU, m)
{
    m.impl("allgather_param", &dc::allgather_param);
    m.impl("prefetch_params_fused", &dc::prefetch_params_fused);
    m.impl("wait_allgather", &dc::wait_allgather);
    m.impl("reduce_grad", &dc::reduce_grad);
    m.impl("free_tensors", &dc::free_tensors);
    m.impl("offload_tensor", &dc::offload_tensor);
    m.impl("reload_tensor", &dc::reload_tensor);
    m.impl("wait_offload", &dc::wait_offload);
    m.impl("wait_reload", &dc::wait_reload);

    m.impl("test_call", &dc::test_call);
}

TORCH_LIBRARY_IMPL(dc, CUDA, m)
{
    m.impl("allgather_param", &dc::allgather_param);
    m.impl("prefetch_params_fused", &dc::prefetch_params_fused);
    m.impl("wait_allgather", &dc::wait_allgather);
    m.impl("reduce_grad", &dc::reduce_grad);
    m.impl("free_tensors", &dc::free_tensors);
    m.impl("offload_tensor", &dc::offload_tensor);
    m.impl("reload_tensor", &dc::reload_tensor);
    m.impl("wait_offload", &dc::wait_offload);
    m.impl("wait_reload", &dc::wait_reload);

    m.impl("test_call", &dc::test_call);
}

TORCH_LIBRARY_IMPL(dc, Meta, m)
{
    m.impl("allgather_param", &dc::allgather_param_meta);
    m.impl("wait_allgather", &dc::wait_allgather_meta);
    m.impl("reduce_grad", &dc::reduce_grad_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("register_z3_param", &dc::register_z3_param, "Register a parameter");
    m.def("set_persistent", &dc::set_persistent, "Set persistent flag for a parameter");
    m.def("enable_profiling", &dc::enable_profiling, "Enable profiling");
    m.def("is_profiling", &dc::is_profiling, "Check if profiling is enabled");
    m.def("init_z3", &dc::init_z3, "Set the process group");
    m.def("cleanup", &dc::cleanup, "Cleanup the process group");
    m.def("register_graph", &dc::register_graph, "Register graph with a list of ds parameter ids");
    m.def("register_graph_ops",
          &dc::register_graph_ops,
          "Register the number of arguments for an op");
    m.def("register_bwd_graph_ops",
          &dc::register_bwd_graph_ops,
          "Register the number of arguments for a backward op");
    m.def("start_forward", &dc::start_forward, "Start forward pass");
    m.def("end_forward", &dc::end_forward, "End forward pass");
    m.def("start_backward", &dc::start_backward, "Start backward pass");
    // m.def("end_backward", &dc::end_backward, "End backward pass");
    m.def("release_param", &dc::release_param, "Release a parameter");
    m.def("cleanup_z3", &dc::cleanup_z3, "Clean up Z3");
    m.def("reset_z3", &dc::reset_z3, "Reset the state");
    m.def("invalidate_gathered_param", &dc::invalidate_gathered_param, "Invalidate gathered param");
    m.def("clear_all_gathered_params", &dc::clear_all_gathered_params, "Clear all gathered params");
}
