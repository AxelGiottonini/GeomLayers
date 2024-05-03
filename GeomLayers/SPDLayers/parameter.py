from typing import Optional

import torch
import torch.nn as nn

class StiefelParameter(nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor]=None,
        requires_grad: bool=True
    ):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return 'Parameter containing:' + self.data.__repr__()
