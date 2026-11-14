"""NumPy golden reference for equal-chunk array splitting."""

import numpy as np
import numpy.typing as npt


def split_tensor(
  x: npt.NDArray[np.float64],
  split_size: int,
  dim: int = -1,
) -> list[npt.NDArray[np.float64]]:
  """Split ``x`` into chunks of length ``split_size`` along axis ``dim``.

  Args:
      x: Input floating-point NumPy array.
      split_size: Chunk width along ``dim``.
      dim: Axis index to split along.

  Returns:
      List of split floating-point sub-arrays.
  """
  axis_len = x.shape[dim]
  indices = list(range(split_size, axis_len, split_size))
  return [np.asarray(part, dtype=np.float64) for part in np.split(x, indices, axis=dim)]
