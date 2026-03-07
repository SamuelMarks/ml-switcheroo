"""Module docstring."""

import numpy as np


def compute_loss(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Manual log-sum-exp for numerical stability
  m = np.max(logits, axis=-1, keepdims=True)
  log_probs = logits - m - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
  batch_size = logits.shape[0]
  # Assuming targets are integer indices
  return -np.sum(log_probs[np.arange(batch_size), targets]) / batch_size
  # </SWITCHEROO_FAILED_TO_TRANS>
