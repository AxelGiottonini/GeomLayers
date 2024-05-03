from functools import cache
from typing import Tuple

import torch
from torch.utils.data import Dataset

class ToyNormCovarianceDataset(Dataset):
    def __init__(
        self,
        n_class: int=20,
        n_item_per_class: int=1000,
        n_features: int=10,
        mean: float=1e9
    ):
        self.n_class = n_class
        self.n_item_per_class = n_item_per_class
        self.n_features = n_features
        self.mean = mean

    def __len__(self):
        return self.n_class * self.n_item_per_class

    @cache
    def __getitem__(self, id: int) -> Tuple[torch.Tensor, int]:
        norms = self.mean + torch.randn(self.n_features) * (1 + id % self.n_class)**4
        norms = norms - torch.mean(norms, dim=-1)
        conorms = norms[:, None] @ norms[None, :]
        return conorms, id % self.n_class
