// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#define USE_C10D_NCCL

#include <stdio.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

namespace dc {

template <typename K, typename V>
static bool hasKey(const std::unordered_map<K, V>& map, const K& key)
{
    return map.find(key) != map.end();
}

template <typename T>
inline std::string to_string(const T& v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

template <typename L>
size_t productDim(const L& dim)
{
    size_t prod = 1;
    for (auto d : dim) { prod *= d; }
    return prod;
}

template <typename T>
std::string join_as_str(const T& v, const char* delim = ",", const size_t maxlen = 0)
{
    std::stringstream ss;

    if (!v.empty()) {
        auto it = v.begin();
        ss << to_string(*it);
        it++;
        for (; it != v.end(); ++it) {
            if (delim) ss << delim;
            ss << to_string(*it);
        }
    }

    std::string s = ss.str();
    if (maxlen > 0 && s.length() > maxlen) { s = s.substr(0, maxlen) + " ..."; }

    return "[" + s + "]";
}

template <typename T>
std::string tensorPtrToString(T* ptr, size_t size, size_t str_len = 100)
{
    std::vector<T> vals;
    for (size_t i = 0; i < size; i++) {
        vals.push_back(*ptr);
        ptr++;
    }
    return join_as_str(vals, ",", str_len);
}

std::string tensorPtrToString(void* ptr,
                              size_t size,
                              c10::ScalarType datatype,
                              size_t max_elem = 20,
                              size_t max_str_len = 100);

std::string tensorToString(const at::Tensor& t, size_t max_elem = 20, size_t max_str_len = 100);

std::string tensorDimToString(const at::Tensor& t);

at::Tensor test_call(at::Tensor param);

extern c10::intrusive_ptr<c10d::ProcessGroup> process_group;
extern c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem;
extern ncclComm_t nccl_comm;
extern bool use_symm_mem;
extern bool profile;
extern bool pre_div_reduce;

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size);
ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type);
void cleanup();

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

class ReduceTask {
public:
    ReduceTask(long ds_id, at::Tensor grad, at::Tensor send_buf)
        : ds_id_(ds_id), grad_(std::move(grad)), send_buf_(std::move(send_buf))
    {
    }

    long getDSId() const { return ds_id_; }
    at::Tensor getSendBuf() const { return send_buf_; }

private:
    long ds_id_;
    at::Tensor grad_;
    at::Tensor send_buf_;
};

class ReduceBucket {
public:
    ReduceBucket(int64_t size, at::ScalarType scalar_type) : size_(size), scalar_type_(scalar_type)
    {
        buffer_ = torch::empty({size}, at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        offset_ = 0;
    }

    int64_t getSize() const { return size_; }
    int64_t getOffset() const { return offset_; }
    at::Tensor getBuffer() const { return buffer_; }
    at::ScalarType getScalarType() const { return scalar_type_; }

    void reserve(int64_t size)
    {
        if (size > size_) {
            buffer_ =
                torch::empty({size}, at::TensorOptions().dtype(scalar_type_).device(at::kCUDA));
            size_ = size;
        }
    }

    at::Tensor allocate(int64_t numel)
    {
        if (offset_ + numel > size_) {
            throw std::runtime_error("Buffer size exceeds the reduce bucket size");
        }

        at::Tensor result = buffer_.index({torch::indexing::Slice(offset_, offset_ + numel)});
        offset_ += numel;
        return result;
    }

    bool shouldFlush(int64_t numel) { return offset_ > 0 && offset_ + numel > size_; }

    void reset() { offset_ = 0; }

private:
    int64_t size_;
    int64_t offset_;
    at::Tensor buffer_;
    at::ScalarType scalar_type_;
};

class DoubleBufferedReduceBucket {
public:
    DoubleBufferedReduceBucket(int64_t initial_bucket_size, bool enable_double_buffer)
        : initial_bucket_size_(initial_bucket_size), enable_double_buffer_(enable_double_buffer)
    {
    }

    void swap(at::ScalarType scalar_type,
              at::cuda::CUDAStream rs_stream,
              at::cuda::CUDAStream copy_stream)
    {
        assert(hasKey(current_buffer_, scalar_type));
        assert(hasKey(current_buffer_events_, scalar_type));

        current_buffer_.at(scalar_type)->reset();
        current_buffer_events_.at(scalar_type)->record(rs_stream);

        if (enable_double_buffer_) {
            assert(hasKey(shadow_buffer_, scalar_type));
            assert(hasKey(shadow_buffer_events_, scalar_type));

            auto tmp = current_buffer_.at(scalar_type);
            current_buffer_[scalar_type] = shadow_buffer_.at(scalar_type);
            shadow_buffer_[scalar_type] = tmp;

            auto tmp_event = current_buffer_events_.at(scalar_type);
            current_buffer_events_[scalar_type] = shadow_buffer_events_.at(scalar_type);
            shadow_buffer_events_[scalar_type] = tmp_event;
        }
    }

    std::shared_ptr<ReduceBucket> getBuffer(at::ScalarType scalar_type)
    {
        if (!hasKey(current_buffer_, scalar_type)) {
            current_buffer_[scalar_type] =
                std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
            current_buffer_events_[scalar_type] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);

            if (enable_double_buffer_) {
                shadow_buffer_[scalar_type] =
                    std::make_shared<ReduceBucket>(initial_bucket_size_, scalar_type);
                shadow_buffer_events_[scalar_type] =
                    std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            }
        }

        return current_buffer_.at(scalar_type);
    }

    std::shared_ptr<at::cuda::CUDAEvent> getEvent(at::ScalarType scalar_type)
    {
        assert(hasKey(current_buffer_events_, scalar_type));
        return current_buffer_events_.at(scalar_type);
    }

    void clear()
    {
        current_buffer_.clear();
        shadow_buffer_.clear();
        current_buffer_events_.clear();
        shadow_buffer_events_.clear();
    }

private:
    int64_t initial_bucket_size_;
    bool enable_double_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> current_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<ReduceBucket>> shadow_buffer_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> current_buffer_events_;
    std::unordered_map<at::ScalarType, std::shared_ptr<at::cuda::CUDAEvent>> shadow_buffer_events_;
};

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
    void setPersistent(bool persistent) { persistent_ = persistent; }
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
        grad_buffer.zero_();
        params_.emplace(ds_id, DSParam(ds_id, ds_shape, ds_tensor, grad_buffer, persistent));
        valid_[ds_id] = false;
    }

    void registerGatheredParam(long ds_id, at::Tensor ds_tensor)
    {
        gathered_params_.emplace(ds_id, ds_tensor);
    }

    void unregisterGatheredParam(long ds_id)
    {
        assert(hasKey(gathered_params_, ds_id));
        gathered_params_.erase(ds_id);
        valid_[ds_id] = false;
    }

    const std::unordered_map<long, DSParam>& getParams() const { return params_; }

    const DSParam& getParam(long ds_id) const { return params_.at(ds_id); }
    const size_t getNumParams() const { return params_.size(); }
    const at::Tensor& getGatheredParam(long ds_id) const
    {
        assert(hasKey(gathered_params_, ds_id));
        return gathered_params_.at(ds_id);
    }
    bool hasGatheredParam(long ds_id) const { return hasKey(gathered_params_, ds_id); }
    void setPersistent(long ds_id, bool persistent) { params_.at(ds_id).setPersistent(persistent); }

    void setValid(long ds_id, bool valid) { valid_[ds_id] = valid; }
    bool isValid(long ds_id) const
    {
        assert(hasKey(valid_, ds_id));
        return valid_.at(ds_id);
    }

private:
    std::unordered_map<long, DSParam> params_;
    std::unordered_map<long, at::Tensor> gathered_params_;
    std::unordered_map<long, bool> valid_;
};

class CustomOpExecutor {
public:
    CustomOpExecutor(c10::intrusive_ptr<c10d::ProcessGroup> process_group,
                     std::shared_ptr<DSParamRegistry> param_registry,
                     std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets,
                     std::vector<long> ds_ids,
                     ncclComm_t nccl_comm,
                     at::cuda::CUDAStream rs_stream,
                     at::cuda::CUDAStream copy_stream,
                     bool pre_div_reduce)
        : process_group_(process_group),
          param_registry_(std::move(param_registry)),
          reduce_buckets_(std::move(reduce_buckets)),
          ds_ids_(std::move(ds_ids)),
          nccl_comm_(nccl_comm),
          rs_stream_(rs_stream),
          copy_stream_(copy_stream),
          pre_div_reduce_(pre_div_reduce)
    {
        for (long ds_id : ds_ids_) {
            has_acc_grad_[ds_id] = false;

            rs_comp_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
            rs_copy_done_events_[ds_id] =
                std::make_shared<at::cuda::CUDAEvent>(cudaEventDisableTiming);
        }
        reduce_counter_ = ds_ids_.size();
    }
    ~CustomOpExecutor() {}

    void registerOpNArgs(const std::string& op_name, long n_args, bool is_backward)
    {
        GraphOpStates& op_states = is_backward ? op_states_bwd_ : op_states_fwd_;
        op_states.registerOpNArgs(op_name, n_args);
    }

    virtual void startForward() {}

    virtual void endForward() {}

    virtual void startBackward(bool update) { param_updated_ = update; }

    virtual void endBackward() {}

    at::Tensor reduceGrad(at::Tensor grad_tensor, long ds_id)
    {
        int world_size = process_group_->getSize();
        const DSParam& param = param_registry_->getParam(ds_id);
        const auto scalar_type = grad_tensor.scalar_type();
        std::shared_ptr<ReduceBucket> reduce_bucket = reduce_buckets_->getBuffer(scalar_type);

        auto comp_stream = at::cuda::getCurrentCUDAStream();

        if (reduce_bucket->shouldFlush(grad_tensor.numel())) {
            flushReduceBucket(scalar_type);

            // reduce_bucket is swapped in flushReduceBucket if double buffering is enabled
            reduce_bucket = reduce_buckets_->getBuffer(scalar_type);
        }

        if (grad_tensor.numel() > reduce_bucket->getSize()) {
            // extend buckets
            at::cuda::stream_synchronize(rs_stream_);
            reduce_bucket->reserve(grad_tensor.numel());
        }

        at::Tensor reduce_in_buffer = reduce_bucket->allocate(grad_tensor.numel());

        // This ensures the order of reduce_scatter -> copy
        // Without this block, copy may start while reduce_scatter is still running
        reduce_buckets_->getEvent(scalar_type)->block(comp_stream);
        auto copy_src = grad_tensor.contiguous().view({-1});
        // keep references to copy src
        reduce_tasks_[scalar_type].emplace_back(ds_id, copy_src, reduce_in_buffer);

        // computation must be done before copy
        rs_comp_done_events_[ds_id]->record(comp_stream);
        rs_comp_done_events_[ds_id]->block(copy_stream_);
        {
            at::cuda::CUDAStreamGuard guard(copy_stream_);
            reduce_in_buffer.copy_(copy_src, true);
            rs_copy_done_events_[ds_id]->record(copy_stream_);
        }

        reduce_counter_--;

        if (reduce_counter_ == 0) {
            flushAllReduceBuckets();

            reduce_counter_ = ds_ids_.size();

            // This synchronization ensures all of reduce calls are done before optimizer's step.
            at::cuda::stream_synchronize(rs_stream_);

            endBackward();
        }

        return at::Tensor();
    }

    bool hasParam(long ds_id) const { return hasKey(has_acc_grad_, ds_id); }

protected:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_;
    std::shared_ptr<DSParamRegistry> param_registry_;
    std::shared_ptr<DoubleBufferedReduceBucket> reduce_buckets_;
    std::vector<long> ds_ids_;
    ncclComm_t nccl_comm_;
    at::cuda::CUDAStream rs_stream_;
    at::cuda::CUDAStream copy_stream_;
    GraphOpStates op_states_fwd_ = GraphOpStates();
    GraphOpStates op_states_bwd_ = GraphOpStates();

    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_comp_done_events_;
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> rs_copy_done_events_;

    size_t reduce_counter_ = 0;
    bool param_updated_ = false;
    std::unordered_map<at::ScalarType, std::vector<ReduceTask>> reduce_tasks_;
    std::unordered_map<long, bool> has_acc_grad_;
    bool pre_div_reduce_;

    void flushReduceBucket(at::ScalarType scalar_type)
    {
        if (!hasKey(reduce_tasks_, scalar_type)) { return; }

        int64_t tmp_recv_numel = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(rs_stream_);

            if (has_acc_grad_.at(t.getDSId())) {
                tmp_recv_numel += param_registry_->getParam(t.getDSId()).getGradBuffer().numel();
            }
        }

        at::Tensor tmp_recv_buf = at::Tensor();
        if (tmp_recv_numel > 0) {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            tmp_recv_buf = torch::empty({tmp_recv_numel},
                                        at::TensorOptions().dtype(scalar_type).device(at::kCUDA));
        }

        ncclGroupStart();
        int64_t offset = 0;
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();

            bool acc_grad = has_acc_grad_.at(t.getDSId());

            if (acc_grad) {
                recv_buf =
                    tmp_recv_buf.index({torch::indexing::Slice(offset, offset + recv_buf.numel())});
            }

            ncclRedOp_t op = pre_div_reduce_ ? ncclSum : ncclAvg;
            if (pre_div_reduce_) {
                at::cuda::CUDAStreamGuard guard(rs_stream_);
                t.getSendBuf().div_(process_group_->getSize());
            }
            ncclResult_t result = ncclReduceScatter(t.getSendBuf().data_ptr(),
                                                    recv_buf.data_ptr(),
                                                    recv_buf.numel(),
                                                    get_nccl_data_type(scalar_type),
                                                    op,
                                                    nccl_comm_,
                                                    rs_stream_);
            if (result != ncclSuccess) { throw std::runtime_error("NCCL ReduceScatter failed"); }

            if (acc_grad) { offset += recv_buf.numel(); }
        }
        ncclGroupEnd();

        {
            at::cuda::CUDAStreamGuard guard(rs_stream_);
            int64_t offset = 0;
            for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
                bool acc_grad = has_acc_grad_.at(t.getDSId());

                if (acc_grad) {
                    auto recv_buf = param_registry_->getParam(t.getDSId()).getGradBuffer();
                    recv_buf.add_(tmp_recv_buf.index(
                        {torch::indexing::Slice(offset, offset + recv_buf.numel())}));
                    offset += recv_buf.numel();
                }
                has_acc_grad_[t.getDSId()] = true;
            }
        }

        reduce_buckets_->swap(scalar_type, rs_stream_, copy_stream_);

        // Not very sure if this is necessary
        // Want to prevent grad tensor from being released before the copy is done
        auto comp_stream = at::cuda::getCurrentCUDAStream();
        for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
            auto copy_done_event = rs_copy_done_events_.at(t.getDSId());
            copy_done_event->block(comp_stream);
        }
        reduce_tasks_[scalar_type].clear();

        if (tmp_recv_numel > 0) { tmp_recv_buf.record_stream(rs_stream_); }
    }

    void flushAllReduceBuckets()
    {
        for (const auto& it : reduce_tasks_) { flushReduceBucket(it.first); }
    }
};

}  // namespace dc
