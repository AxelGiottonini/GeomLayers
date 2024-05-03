import torch
import torch.nn as nn

from GeomLayers.SPDLayers.parameter import StiefelParameter

class SPDBiMap(nn.Module):
    """
    Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning.
    """
    def __init__(
        self,
        input_features: int,
        output_features: int
    ):
        super().__init__()

        self.weight = StiefelParameter(torch.empty(output_features, input_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        output = torch.einsum("bij,oj->bio", input, self.weight)
        output = torch.einsum("bji,oj->boi", output, self.weight)
        return output
