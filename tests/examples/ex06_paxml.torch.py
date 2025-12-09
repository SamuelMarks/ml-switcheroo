import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
  """
  A simple Multi-Layer Perceptron for PaxML conversion testing.
  Semantic pivot:
    - nn.Module -> praxis.base_layer.BaseLayer
    - __init__ -> setup
    - forward -> __call__
    - nn.Linear -> praxis.layers.Linear
  """

  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    # Standard Linear layer
    self.fc1 = nn.Linear(input_size, hidden_size)
    # Activation
    self.relu = nn.ReLU()
    # Output layer
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
