# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import operator
from typing import List

import torch
from torch.fx import Node, Graph


def get_param_nodes(graph: Graph, n_params: int) -> List[Node]:
    return [n for n in graph.nodes if n.op == "placeholder"][:n_params]


def is_comm_op(node: Node) -> bool:
    return "comm" in node.meta and node.meta["comm"]


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
    elif dtype == torch.float16 or dtype == torch.bfloat16:
        elem_size = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return numel * elem_size
