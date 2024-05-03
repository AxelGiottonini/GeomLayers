import torch

def symmetric(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * (input + input.transpose(-1, -2))

