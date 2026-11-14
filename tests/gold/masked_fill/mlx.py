"""Apple MLX golden reference for attention score masked fill."""

import mlx.core as mx


def causal_mask_fill(
  scores: mx.array,
  mask: mx.array,
  value: float = -1000000000.0,
) -> mx.array:
  """Replace positions where ``mask == 0`` with ``value`` using ``mx.where``.

  Args:
      scores: Unmasked attention score array.
      mask: Binary or boolean mask array (``0`` indicates masked entries).
      value: Fill constant substituted at masked positions.

  Returns:
      Masked attention score array.
  """
  res: mx.array = mx.where(mask == 0, value, scores)
  return res
