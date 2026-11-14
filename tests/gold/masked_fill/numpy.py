"""NumPy golden reference for attention score masked fill."""

import numpy as np
import numpy.typing as npt


def causal_mask_fill(
  scores: npt.NDArray[np.float64],
  mask: npt.NDArray[np.int64],
  value: float = -1000000000.0,
) -> npt.NDArray[np.float64]:
  """Replace positions where ``mask == 0`` with ``value`` using ``np.where``.

  Args:
      scores: Unmasked floating-point score array.
      mask: Integer mask array where ``0`` marks masked positions.
      value: Fill value placed at masked indices.

  Returns:
      Masked floating-point score array.
  """
  return np.asarray(np.where(mask == 0, value, scores), dtype=np.float64)
