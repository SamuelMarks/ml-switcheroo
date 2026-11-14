"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
import optax


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> jnp.ndarray:
  """Gets clipped optimizer.

  Args:
      learning_rate: Input `learning_rate` parameter value (`float`).
      max_norm: Input `max_norm` parameter value (`float`).

  Returns:
      Resulting `jnp.ndarray` from get clipped optimizer.
  """
  return optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(learning_rate))


def clip_grads(grads: jnp.ndarray, max_norm: float = 1.0) -> jnp.ndarray:
  """Clip gradients by global norm using ``optax.clip_by_global_norm``.

  Args:
      grads: Input JAX gradient array or PyTree leaf.
      max_norm: Maximum global L2 norm threshold.

  Returns:
      Clipped JAX gradient array.
  """
  tx = optax.clip_by_global_norm(max_norm)
  clipped, _ = tx.update(grads, tx.init(grads))
  res: jnp.ndarray = clipped
  return res
