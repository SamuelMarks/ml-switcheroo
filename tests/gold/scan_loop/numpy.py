"""NumPy golden reference for sequential recurrent scan loops."""

import typing

import numpy as np
import numpy.typing as npt


def rnn_loop(
  cell: typing.Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
  ],
  x: npt.NDArray[np.float64],
  init_state: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  """Iterate ``cell`` across leading time dimension 0 of ``x`` in NumPy.

  Args:
      cell: Recurrent step function ``(x_t, state) -> (y_t, next_state)``.
      x: Time-major input floating-point array.
      init_state: Initial state floating-point array.

  Returns:
      Tuple of ``(stacked_outputs, final_state)``.
  """
  outputs: list[npt.NDArray[np.float64]] = []
  state = init_state
  for idx in range(x.shape[0]):
    out, state = cell(x[idx], state)
    outputs.append(out)
  return np.asarray(np.stack(outputs, axis=0), dtype=np.float64), state
