"""Apple MLX golden reference for global-norm gradient clipping."""

import mlx.core as mx
import mlx.optimizers as optim


def clip_grads(
  grads: dict[str, mx.array],
  max_norm: float = 1.0,
) -> tuple[dict[str, mx.array], mx.array]:
  """Clip a parameter gradient tree by global L2 norm using ``mlx.optimizers.clip_grad_norm``.

  Args:
      grads: Gradient dictionary keyed by parameter path.
      max_norm: Maximum global L2 norm threshold.

  Returns:
      Tuple of ``(clipped_grads, total_norm)``.
  """
  clipped, total_norm = optim.clip_grad_norm(grads, max_norm=max_norm)
  return clipped, total_norm
