# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import operator

import torch
from torch.fx import Node


def is_comm_op(node: Node) -> bool:
    return hasattr(node.meta, "comm") and node.meta["comm"]


def dtype_to_elem_size(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float64:
        elem_size = 8
    elif dtype == torch.float16:
        elem_size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return elem_size


def tensor_meta_size(tensor_meta) -> int:
    numel = functools.reduce(operator.mul, tensor_meta.shape)

    dtype = tensor_meta.dtype
    if dtype == torch.float32:
        elem_size = 4
    elif dtype == torch.float64:
        elem_size = 8
    elif dtype == torch.float16:
        elem_size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return numel * elem_size
