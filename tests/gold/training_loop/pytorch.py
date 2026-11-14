"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
import torch.optim as optim


def train_step(
  model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, y: torch.Tensor, loss_fn: torch.Tensor
) -> torch.Tensor:
  """Trains step.

  Args:
      model: Neural network model instance under test.
      optimizer: Input `optimizer` parameter value (`optim.Optimizer`).
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: Input `loss_fn` parameter value (`torch.Tensor`).

  Returns:
      Resulting `torch.Tensor` from train step.
  """
  model.train()
  optimizer.zero_grad()
  predictions = model(x)
  loss = loss_fn(predictions, y)
  loss.backward()
  optimizer.step()
  res: torch.Tensor = loss
  return res
