"""Module docstring."""

import numpy as np


def train_step(model, optimizer, x: np.ndarray, y: np.ndarray, loss_fn):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Complete manual backpropagation required
  # Example is structurally omitted due to complexity
  predictions = model(x)
  loss = loss_fn(predictions, y)
  # Manual backprop...
  # grads = compute_grads(model, predictions, y)
  # optimizer.step(model, grads)
  return loss
  # </SWITCHEROO_FAILED_TO_TRANS>
