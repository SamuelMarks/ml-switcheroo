"""PaxML / JAX golden reference for functional ``jax.lax.scan`` recurrent loops."""

import typing

import jax


def rnn_loop(
  cell: typing.Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
  x: jax.Array,
  init_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Scan ``cell`` across ``x`` functionally via ``jax.lax.scan``.

  Args:
      cell: Step callable ``(x_t, state) -> (y_t, next_state)``.
      x: Time-major input JAX array.
      init_state: Initial carry state JAX array.

  Returns:
      Tuple of ``(stacked_outputs, final_state)``.
  """

  def _scan_body(carry: jax.Array, x_t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Adapt ``cell(x_t, carry)`` into ``jax.lax.scan`` ``(carry, y_t)`` order.

    Args:
        carry: Recurrent carry state.
        x_t: Current time-step input slice.

    Returns:
        Updated ``(next_carry, step_output)`` pair.
    """
    out, next_carry = cell(x_t, carry)
    return next_carry, out

  final_state, outputs = jax.lax.scan(_scan_body, init_state, x)
  return outputs, final_state
