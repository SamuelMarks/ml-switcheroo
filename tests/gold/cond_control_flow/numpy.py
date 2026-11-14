"""NumPy golden reference for data-dependent conditional control flow."""

import numpy as np
import numpy.typing as npt


def conditional_op(
  pred: npt.NDArray[np.bool_],
  x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Select ``x * 2`` when ``pred`` is True and ``x + 2`` otherwise.

  Args:
      pred: Boolean scalar or broadcastable mask array.
      x: Input floating-point array operand.

  Returns:
      Selected floating-point NumPy array.
  """
  if bool(np.all(pred)):
    return np.asarray(x * 2.0, dtype=np.float64)
  return np.asarray(x + 2.0, dtype=np.float64)
