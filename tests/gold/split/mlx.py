"""Apple MLX golden reference for equal-chunk tensor splitting."""

import mlx.core as mx


def split_tensor(x: mx.array, split_size: int, dim: int = -1) -> list[mx.array]:
  """Split ``x`` into chunks of size ``split_size`` along ``dim`` in MLX.

  Args:
      x: Input MLX array to partition.
      split_size: Size of each chunk along ``dim``.
      dim: Axis along which to split.

  Returns:
      List of split MLX sub-arrays.
  """
  num_sections = x.shape[dim] // split_size
  parts: list[mx.array] = list(mx.split(x, num_sections, axis=dim))
  return parts
