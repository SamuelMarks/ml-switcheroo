"""Test suite for the Numpy module."""

import numpy as np


def train_step(model, optimizer, x: np.ndarray, y: np.ndarray, loss_fn):
  """Trains step."""
  predictions = model(x)
  loss = loss_fn(predictions, y)
  return loss
