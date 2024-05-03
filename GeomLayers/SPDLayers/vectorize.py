import torch
import torch.nn as nn

class SPDVectorize(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        row_idx, col_idx = torch.tril_indices(input_features, input_features)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input[:, self.row_idx, self.col_idx]
        return output
