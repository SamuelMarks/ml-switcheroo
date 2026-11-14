"""NumPy golden reference for global-norm gradient clipping."""

import numpy as np
import numpy.typing as npt


def clip_grads(
  grads: list[npt.NDArray[np.float64]],
  max_norm: float = 1.0,
) -> list[npt.NDArray[np.float64]]:
  """Clip a list of NumPy gradient arrays by their joint global L2 norm.

  Args:
      grads: List of gradient arrays to rescale.
      max_norm: Maximum permitted global L2 norm.

  Returns:
      Rescaled gradient arrays whose joint L2 norm is at most ``max_norm``.
  """
  sq_sum = sum(float(np.sum(np.square(g))) for g in grads)
  total_norm = float(np.sqrt(sq_sum))
  if total_norm > max_norm and total_norm > 0.0:
    scale = max_norm / total_norm
    return [np.asarray(g * scale, dtype=np.float64) for g in grads]
  return [np.asarray(g, dtype=np.float64) for g in grads]
