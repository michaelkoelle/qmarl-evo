"""class for neuralnetwork as agent"""

# Workaround because Any != Any
# pyright: reportIncompatibleMethodOverride=false

import torch.nn.functional as F
from torch import Tensor, nn


# nr_intput_features == obsspace, nr_hidden_units soll 64 sein
class NeuralNetwork(nn.Module):
    """Neural Network"""

    def __init__(self, input_dimension: int, nr_actions: int):
        super(NeuralNetwork, self).__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(input_dimension, 2),
            nn.ELU(),
            nn.Linear(2, 1),
            nn.ELU(),
            nn.Linear(1, nr_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward"""
        x = self.fc_net(x)
        x = F.softmax(x, dim=-1)
        return x
