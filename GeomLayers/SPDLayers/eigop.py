from typing import Any, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Function

from GeomLayers.SPDLayers.utils import symmetric

__all__ = [
    "SPDExpEig",
    "SPDLogEig",
    "SPDReEig",
    "SPDTopReEig"
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
    output_s = operation.fn(s, *args)
    output = u @ torch.diag_embed(output_s) @ u.transpose(-2, -1)
    return output, u, s, output_s

def eigbackward(
    gradient: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    output_s: torch.Tensor,
    operation: Type[EigOp],
    *args
) -> torch.Tensor:
    ds=torch.diag_embed(operation.d(s, *args))
    L = (output_s[:, :, None] - output_s[:, None, :]) / (s[:, :, None] - s[:, None, :])
    L[torch.logical_or(torch.isnan(L), torch.isinf(L))] = 0
    L = L + ds
    dp = L * (u.transpose(-1, -2) @ gradient @ u)
    dp = u @ dp @ u.transpose(-1, -2)

    return dp

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
        self.register_buffer("epsilon", torch.Tensor([epsilon]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDReEigFunction.apply(input, self.epsilon[0])

class TopReEigOp(EigOp):
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    @classmethod
    def fn(cls, s: torch.Tensor, n: int) -> torch.Tensor:
        threshold = s[:, n][:, None]
        return (s >= threshold) * s + (s < threshold) * threshold

    @classmethod
    def d(cls, s: torch.Tensor, n: int) -> torch.Tensor:
        threshold = s[:, n][:, None]
        return (s >= threshold)

class SPDTopReEigOpFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, n: int) -> torch.Tensor:
        output, u, s, output_s = eigforward(input, TopReEigOp, n)
        ctx.save_for_backward(u, s, output_s, n)
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        u, s, output_s, n = ctx.saved_tensors
        output_gradient = eigbackward(gradient, u, s, output_s, TopReEigOp, n)
        return output_gradient, None

class SPDTopReEig(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("n", torch.LongTensor([n]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SPDTopReEigOpFunction.apply(input, self.n[0])