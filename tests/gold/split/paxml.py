"""PaxML / JAX golden reference for equal-chunk tensor splitting."""

import jax.numpy as jnp


def split_tensor(x: jnp.ndarray, split_size: int, dim: int = -1) -> list[jnp.ndarray]:
  """Split ``x`` into chunks of size ``split_size`` along ``dim`` via ``jnp.split``.

  Args:
      x: Input JAX array to split.
      split_size: Chunk size along ``dim``.
      dim: Axis along which to split.

  Returns:
      List of split JAX sub-arrays.
  """
  indices = list(range(split_size, x.shape[dim], split_size))
  parts: list[jnp.ndarray] = list(jnp.split(x, indices, axis=dim))
  return parts
