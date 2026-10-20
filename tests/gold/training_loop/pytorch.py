"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
import torch.optim as optim


def train_step(
  model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, y: torch.Tensor, loss_fn: torch.Tensor
) -> torch.Tensor:
  """Trains step.

  Args:
      model: Test argument or fixture.
      optimizer: Test argument or fixture.
      x: Test argument or fixture.
      y: Test argument or fixture.
      loss_fn: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  model.train()
  optimizer.zero_grad()
  predictions = model(x)
  loss = loss_fn(predictions, y)
  loss.backward()
  optimizer.step()
  res: torch.Tensor = loss
  return res
