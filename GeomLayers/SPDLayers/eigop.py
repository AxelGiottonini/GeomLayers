from typing import Any, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Function

from GeomLayers.SPDLayers.utils import symmetric

__all__ = [
    "SPDExpEig",
    "SPDLogEig",
    "SPDReEig"
]

class EigOp():
    @classmethod
    def fn(cls, s: torch.Tensor, *args) -> torch.Tensor:
        pass

    @classmethod
    def d(cls, s: torch.Tensor, *args) -> torch.Tensor:
        pass

def eigforward(
    input: torch.Tensor,
    operation: Type[EigOp],
    *args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, _ = torch.linalg.svd(input)
    output_s = torch.diag_embed(operation.fn(s, *args))
    output = u @ output_s @ u.transpose(-2, -1)
    return output, u, s, output_s

def eigbackward(
    gradient: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    output_s: torch.Tensor,
    operation: Type[EigOp],
    *args
) -> torch.Tensor:
    gradient = symmetric(gradient)
    eye = torch.eye(gradient.shape[-1], dtype=gradient.dtype, device=gradient.device)
    ds = torch.diag_embed(operation.d(s, *args))
    dLdV = 2 * gradient @ u @ output_s
    dLdS = eye * (ds @ u.transpose(-1, -2) @ gradient @ u)
    p = 1 / (s[:, :, None] - s[:, None, :] + 1e-6)
    p = (1 - eye) * p + eye * 0
    return u @ (symmetric(p.transpose(-1, -2) * (u.transpose(-1, -2) @ dLdV)) + dLdS) @ u.transpose(-1, -2)

class ExpEigOp(EigOp):
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    @classmethod
    def fn(cls, s: torch.Tensor) -> torch.Tensor:
        return torch.exp(s)

    @classmethod
    def d(cls, s: torch.Tensor) -> torch.Tensor:
        return torch.exp(s)

class SPDExpEigFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output, u, s, output_s = eigforward(input, ExpEigOp)
        ctx.save_for_backward(u, s, output_s)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        u, s, output_s = ctx.saved_tensors
        output_gradient = eigbackward(gradient, u, s, output_s, ExpEigOp)
        return output_gradient

class SPDExpEig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDExpEigFunction.apply(input)

class LogEigOp(EigOp):
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    @classmethod
    def fn(cls, s: torch.Tensor) -> torch.Tensor:
        return torch.log(s + 1e-6)

    @classmethod
    def d(cls, s: torch.Tensor) -> torch.Tensor:
        return 1 / (s + 1e-6)

class SPDLogEigFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output, u, s, output_s = eigforward(input, LogEigOp)
        ctx.save_for_backward(u, s, output_s)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        u, s, output_s = ctx.saved_tensors
        output_gradient = eigbackward(gradient, u, s, output_s, LogEigOp)
        return output_gradient

class SPDLogEig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDLogEigFunction.apply(input)

class ReEigOp(EigOp):
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    @classmethod
    def fn(cls, s: torch.Tensor, epsilon: float) -> torch.Tensor:
        return f.threshold(s, epsilon, epsilon)

    @classmethod
    def d(cls, s: torch.Tensor, epsilon: float) -> torch.Tensor:
        return (s > epsilon).float()

class SPDReEigFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, epsilon: float) -> torch.Tensor:
        output, u, s, output_s = eigforward(input, ReEigOp, epsilon)
        ctx.save_for_backward(u, s, output_s, epsilon)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        u, s, output_s, epsilon = ctx.saved_tensors
        output_gradient = eigbackward(gradient, u, s, output_s, ReEigOp, epsilon)
        return output_gradient, None

class SPDReEig(nn.Module):
    def __init__(self, epsilon: float=1e-4):
        super().__init__()
        self.register_buffer("epsilon", torch.LongTensor([epsilon]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDReEigFunction.apply(input, self.epsilon[0])