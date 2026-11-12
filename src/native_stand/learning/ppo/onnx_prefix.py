"""把显式Gemm/ELU/Tanh ONNX转成只用于前控的冻结Torch模型。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def from_onnx(model):
    from onnx import helper, numpy_helper

    arrays = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if (
                attrs.get("alpha", 1.0) != 1
                or attrs.get("beta", 1.0) != 1
                or attrs.get("transA", 0) != 0
            ):
                raise ValueError("不支持带额外缩放的ONNX Gemm")
            weight = arrays[node.input[1]]
            weight = weight if attrs.get("transB", 0) else weight.T
            bias = arrays[node.input[2]]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(np.array(weight, copy=True)))
                layer.bias.copy_(torch.from_numpy(np.array(bias, copy=True)))
            layers.append(layer)
        elif node.op_type == "Elu":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            layers.append(nn.ELU(alpha=float(attrs.get("alpha", 1.0))))
        elif node.op_type == "Tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError("前控ONNX含未声明操作: " + node.op_type)
    if not layers or not isinstance(layers[-1], nn.Tanh):
        raise ValueError("前控ONNX必须显式以Tanh结束")
    sequence = nn.Sequential(*layers).eval().requires_grad_(False)

    class Prefix(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence = sequence

        def mean(self, value):
            return self.sequence(value)

    return Prefix().eval().requires_grad_(False)
