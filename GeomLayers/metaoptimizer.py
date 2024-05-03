import torch.nn as nn

from GeomLayers.SPDLayers import StiefelParameter
from GeomLayers.SPDLayers.optimizer import StiefelOptimizer

class MetaOptimizer():
    def __init__(self, parameters: nn.ParameterList, base_optimizer, lr=1e-2, *args, **kwargs):
        parameters = list(parameters)
        parameters = [p for p in parameters if p.requires_grad]

        self.parameters = {
            "base": [p for p in parameters if type(p) == nn.Parameter],
            "stiefel": [p for p in parameters if type(p) == StiefelParameter]
        }

        self.optimizers = {
            "base": base_optimizer(self.parameters["base"], lr, *args, **kwargs),
            "stiefel": StiefelOptimizer(self.parameters["stiefel"], lr)
        }

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

