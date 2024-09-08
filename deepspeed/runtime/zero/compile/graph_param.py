# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass, field
from typing import Any, Dict, List
from functools import reduce

import torch
from torch.fx import Graph, Node

from .fx import get_output_node


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

        self._param_nodes = [n for n in self._fw_graph.nodes if n.op == "placeholder"][:len(ds_ids)]
        param_inputs = sample_inputs[:len(ds_ids)]

        for pn, pi, ds_id in zip(self.param_nodes, param_inputs, ds_ids):
            self._params[pn.name] = DSGraphParam(name=pn.name,
                                                 shape=pi.size(),
                                                 dtype=pi.dtype,
                                                 device=pi.device,
                                                 node=pn,
                                                 allgather_node=None,
                                                 release_node=None,
                                                 param=pi)
            self._ds_ids[pn.name] = ds_id

    def add_bw_graph(self, bw_graph: Graph):
        self._bw_graph = bw_graph

        output_node = get_output_node(self._bw_graph)
        self._param_nodes_bw = [n for n in self._bw_graph.nodes if n.name in self.param_names]
        self._param_name_to_grad = {
            param_node.name: grad
            for param_node, grad in zip(self.param_nodes, output_node.args[0])
        }

    @property
    def param_nodes(self):
        return self._param_nodes

    @property
    def param_names(self):
        return [pn.name for pn in self.param_nodes]

    @property
    def ds_ids(self):
        return self._ds_ids

    @property
    def param_nodes_bw(self):
        return self._param_nodes_bw

    def get_input_nodes(self, bw=False):
        graph = self._bw_graph if bw else self._fw_graph
        return [n for n in graph.nodes if n.op == "placeholder"]

    def get_graph_param(self, param_name):
        return self._params[param_name]

    def get_grad_name(self, param_name):
        assert self._param_name_to_grad is not None, "Backward graph is not added yet"
        return self._param_name_to_grad[param_name]
