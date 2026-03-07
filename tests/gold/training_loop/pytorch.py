"""Module docstring."""

import torch
import torch.nn as nn
import torch.optim as optim


def train_step(model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, y: torch.Tensor, loss_fn) -> torch.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Imperative, in-place gradients
  model.train()
  optimizer.zero_grad()
  predictions = model(x)
  loss = loss_fn(predictions, y)
  loss.backward()
  optimizer.step()
  return loss
  # </SWITCHEROO_FAILED_TO_TRANS>
