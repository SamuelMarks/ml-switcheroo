"""Apple MLX golden reference for sequential recurrent scan loops."""

import typing

import mlx.core as mx


def rnn_loop(
  cell: typing.Callable[[mx.array, mx.array], tuple[mx.array, mx.array]],
  x: mx.array,
  init_state: mx.array,
) -> tuple[mx.array, mx.array]:
  """Step a recurrent cell across time dimension 0 of ``x`` in MLX.

  Args:
      cell: Recurrent step function ``(x_t, state) -> (y_t, next_state)``.
      x: Time-major input sequence array.
      init_state: Initial hidden state array.

  Returns:
      Tuple of ``(stacked_outputs, final_state)``.
  """
  outputs: list[mx.array] = []
  state = init_state
  for idx in range(x.shape[0]):
    out, state = cell(x[idx], state)
    outputs.append(out)
  stacked: mx.array = mx.stack(outputs, axis=0)
  return stacked, state
