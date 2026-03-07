"""Module docstring."""

import torch
import torch.nn as nn
import torch.optim as optim


def setup_adam(model: nn.Module, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # PyTorch optimizers update parameters in-place
  return optim.Adam(model.parameters(), lr=lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
