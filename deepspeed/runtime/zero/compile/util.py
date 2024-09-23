# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import operator
from typing import List, Tuple

import torch
from torch.fx import Node, Graph


def get_input_nodes(graph: Graph) -> List[Node]:
    return [n for n in graph.nodes if n.op == "placeholder"]


def get_param_nodes(graph: Graph, index_to_ds_ids: List[Tuple[int, int]]) -> List[Node]:
    all_input_nodes = get_input_nodes(graph)
    return [all_input_nodes[i] for i, _, _ in index_to_ds_ids]


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
