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
void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    bool persistent);

extern c10::intrusive_ptr<c10d::ProcessGroup> process_group;
extern c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem;
extern ncclComm_t nccl_comm;
extern bool use_symm_mem;
extern bool profile;
extern bool pre_div_reduce;

c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> getSymmMemWorkspace(int64_t size);
ncclDataType_t get_nccl_data_type(at::ScalarType scalar_type);

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

}  // namespace dc
