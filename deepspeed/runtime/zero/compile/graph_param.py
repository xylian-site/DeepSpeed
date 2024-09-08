# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, field
from typing import Any, Dict, List
from functools import reduce

import torch
from torch.fx import Graph, Node

from .fx import get_output_node
from .util import get_param_nodes


@dataclass
class DSGraphParam:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    node: Node
    allgather_node: Node
    release_node: Node
    param: torch.Tensor
    numel: int = field(init=False)

    def __post_init__(self):
        self.numel = reduce(lambda x, y: x * y, self.shape)


class DSGraphParamManager:

    def __init__(self, fw_graph: Graph, sample_inputs: Any, ds_ids: List[int]):
        self._fw_graph = fw_graph
        self._bw_graph = None
        self._params: Dict[str, DSGraphParam] = {}
        self._param_name_to_grad: Dict[str, Node] = {}
        self._ds_ids: Dict[str, int] = {}

        param_nodes = get_param_nodes(fw_graph, len(ds_ids))
        self._param_names = [pn.name for pn in param_nodes]

        param_inputs = sample_inputs[:len(ds_ids)]

        for pn, pi, ds_id in zip(param_nodes, param_inputs, ds_ids):
            self._params[pn.name] = DSGraphParam(name=pn.name,
                                                 shape=pi.size(),
                                                 dtype=pi.dtype,
                                                 device=pi.device,
                                                 node=pn,
                                                 allgather_node=None,
                                                 release_node=None,
                                                 param=pi)
            self._ds_ids[pn.name] = ds_id

    def get_bwd_mapping(self, bw_graph: Graph):
        self._bw_graph = bw_graph

        output_node = get_output_node(bw_graph)
        param_nodes_bw = [n for n in self._bw_graph.nodes if n.name in self.param_names]
        param_name_to_grad = {param_name: grad for param_name, grad in zip(self.param_names, output_node.args[0])}
        return param_nodes_bw, param_name_to_grad

    @property
    def param_names(self):
        return self._param_names

    @property
    def ds_ids(self):
        return self._ds_ids

    def get_grad_name(self, param_name):
        assert self._param_name_to_grad is not None, "Backward graph is not added yet"
        return self._param_name_to_grad[param_name]
