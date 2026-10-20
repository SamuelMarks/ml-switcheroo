"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


def compute_loss(
  logits: np.typing.NDArray[np.float64], targets: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:
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
  return np.asarray(-np.sum(log_probs[np.arange(batch_size), targets]) / batch_size, dtype=np.float64)
