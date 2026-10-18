"""Test suite for the Numpy module."""

import typing

import numpy as np


def compute_loss(logits: np.ndarray[typing.Any, typing.Any], targets: np.ndarray[typing.Any, typing.Any]) -> typing.Any:
  """Computes loss.

  Args:
      logits: Test argument or fixture.
      targets: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  m = np.max(logits, axis=-1, keepdims=True)
  log_probs = logits - m - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
  batch_size = logits.shape[0]
  return -np.sum(log_probs[np.arange(batch_size), targets]) / batch_size
