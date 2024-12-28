// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#pragma once

namespace dc {

void register_graph_z3(long graph_id, const std::vector<long>& ds_ids);
void register_graph_ops_z3(long graph_id,
                           const std::vector<std::string>& op_names,
                           const std::vector<long>& n_args);
void register_bwd_graph_ops_z3(long graph_id,
                               const std::vector<std::string>& op_names,
                               const std::vector<long>& n_args);
void init_z3(c10::intrusive_ptr<c10d::ProcessGroup> pg,
             int64_t initial_reduce_bucket_size,
             bool enable_double_buffer,
             bool _use_symm_mem);
void reset_z3();
void cleanup_z3();
void register_z3_param(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       bool persistent);
at::Tensor allgather_param(at::Tensor param_tensor, long graph_id, long ds_id);
void set_persistent(long ds_id);
void prefetch_params_fused(long graph_id,
                           const std::vector<at::Tensor> params,
                           const std::vector<long>& ds_ids);
// for profiling
void invalidate_gathered_param(long ds_id);
void clear_all_gathered_params();
at::Tensor allgather_param_meta(at::Tensor param_tensor, long graph_id, long ds_id);
void release_param(long graph_id, long ds_id);
at::Tensor wait_allgather(at::Tensor v,
                          long graph_id,
                          const std::vector<long>& ds_ids,
                          const std::string& user,
                          long n_args,
                          bool is_backward);
at::Tensor wait_allgather_meta(at::Tensor v,
                               long graph_id,
                               const std::vector<long>& ds_ids,
                               const std::string& user,
                               long n_args,
                               bool is_backward);
at::Tensor offload_tensor(at::Tensor tensor, long graph_id, long id);
at::Tensor reload_tensor(at::Tensor tensor, long graph_id, long id);
at::Tensor wait_offload(at::Tensor tensor, long graph_id, long id);
at::Tensor wait_reload(at::Tensor tensor, long graph_id, long id);
void start_forward();
void end_forward();
void start_backward(bool update);
}  // namespace dc
