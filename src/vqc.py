"""Definition of Variational Quantum Circuit"""
# Workaround because Any != Any
# pyright: reportIncompatibleMethodOverride=false

import numpy as np
import pennylane as qml
import torch
from torch import Tensor, nn


# Help Functions: QC-Operations
def rotate(w_layer: Tensor):
    """Rotations"""
    for i, w_wire in enumerate(w_layer):
        theta_x, theta_y, theta_z = w_wire
        qml.Rot(theta_x, theta_y, theta_z, wires=i)


def entangle(n_qubits: int):
    """Entangles the qubits"""
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    qml.CNOT(wires=[n_qubits - 1, 0])


def encode(state: Tensor):
    """Encodes the state"""
    for i, s in enumerate(state):
        qml.RX(np.arctan(s), wires=i)


# VQC circuit
class VQC(nn.Module):
    """Variational quantum circuit"""

    # action space depends on env
    def __init__(self, num_qubits: int, num_layers: int, action_space: int):
        super().__init__()

        # layers and qubits
        self.num_layers = num_layers
        self.num_qubits = num_qubits

        # action space dimension
        self.action_space = action_space

        # create qnode with choosen device
        self.device = qml.device("default.qubit", wires=range(num_qubits))
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        # start parameters random calculated
        self.weights = nn.Parameter(
            (torch.rand(size=(self.num_layers, self.num_qubits, 3)) * 2 * torch.pi) - torch.pi,
            requires_grad=False,
        )
        # bias
        # self.bias = nn.Parameter((torch.rand(self.action_space) * 0.01), requires_grad=False)
        self.bias = nn.Parameter((torch.ones(self.action_space) * 666), requires_grad=False)

    # using x for states
    def circuit(
        self, weights: Tensor, x: Tensor
    ):  # for different encoding you might need x: Tensor
        """Builds the circuit"""
        # encode(x)
        qml.AmplitudeEmbedding(
            features=x, wires=range(self.num_qubits), pad_with=0.3, normalize=True
        )
        for i in range(self.num_layers):
            entangle(self.num_qubits)

            rotate(weights[i])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return torch.stack([self.qnode(self.weights, x_i) + self.bias for x_i in x])

    def show(self, x: Tensor):
        """Prints the circuit"""
        draw = qml.draw(self.qnode)
        print(draw(self.weights, x))
