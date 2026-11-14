"""PaxML / Optax golden reference for global-norm gradient clipping."""

import jax
import optax


def clip_grads(grads: dict[str, jax.Array], max_norm: float = 1.0) -> dict[str, jax.Array]:
  """Clip a gradient PyTree by global norm via ``optax.clip_by_global_norm``.

  Args:
      grads: PyTree mapping parameter names to JAX gradient arrays.
      max_norm: Maximum global norm threshold.

  Returns:
      Clipped gradient PyTree dictionary.
  """
  tx = optax.clip_by_global_norm(max_norm)
  clipped, _ = tx.update(grads, tx.init(grads))
  res: dict[str, jax.Array] = clipped
  return res
