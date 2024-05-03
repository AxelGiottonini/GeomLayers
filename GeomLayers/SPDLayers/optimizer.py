from typing import NoReturn

import torch
import torch.nn as nn

from GeomLayers.SPDLayers.parameter import StiefelParameter

class StiefelOptimizer():
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    def __init__(self, parameters: nn.Parameter, lr: float):
        self.parameters = parameters
        self.lr = lr
        self.state = {}

    def zero_grad(self) -> NoReturn:
        for p in self.parameters:
            if not isinstance(p, StiefelParameter):
                continue

            if p.grad is None:
                continue

            p.grad.data.zero_()

    def step(self) -> NoReturn:
        for p in self.parameters:
            if not isinstance(p, StiefelParameter):
                continue

            if p.grad is None and p.requires_grad:
                continue

            self.state[id(p)] = p.data
            direction = stiefel_projection(p.grad.data, p.data)
            p.data = stiefel_retraction(-self.lr * direction, p.data)

def stiefel_projection(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    out = input1 - input2 @ input1.transpose(-1, -2) @ input2
    return out

def stiefel_retraction(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    A = input1 + input2
    Q, R = torch.linalg.qr(A)
    out = Q @ torch.sign(R)
    return out