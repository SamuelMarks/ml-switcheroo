import torch.nn as nn


class SimplePerceptron(nn.Module):
  """
  Basic Single-Layer Perceptron.
  Semantic pivot:
    - nn.Module -> flax.nnx.Module
    - nn.Linear -> flax.nnx.Linear
    - forward -> __call__
  """

  def __init__(self, in_features, out_features):
    super().__init__()
    # Standard Linear layer
    self.layer = nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.layer(x)
