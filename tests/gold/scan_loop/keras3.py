"""Keras 3 golden reference for sequential recurrent scan loops."""

import typing

import keras


def rnn_loop(
  cell: typing.Callable[[keras.KerasTensor, keras.KerasTensor], tuple[keras.KerasTensor, keras.KerasTensor]],
  x: keras.KerasTensor,
  init_state: keras.KerasTensor,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
  """Unroll a recurrent step function over the leading time axis in Keras 3.

  Args:
      cell: Step callable accepting ``(step_input, state)`` and returning ``(out, next_state)``.
      x: Time-major input sequence tensor.
      init_state: Initial recurrent state tensor.

  Returns:
      Tuple of ``(stacked_outputs, final_state)``.
  """
  outputs: list[keras.KerasTensor] = []
  state = init_state
  steps = int(x.shape[0])
  for idx in range(steps):
    out, state = cell(x[idx], state)
    outputs.append(out)
  stacked: keras.KerasTensor = keras.ops.stack(outputs, axis=0)
  return stacked, state
