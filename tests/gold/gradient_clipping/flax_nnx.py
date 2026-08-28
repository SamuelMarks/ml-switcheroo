"""Test suite for the Flax Nnx module."""

import typing
import optax  # type: ignore


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> typing.Any:
  """Gets clipped optimizer."""
  return optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))
