"""Flax NNX golden reference for Weight-Normalized Linear projection."""

import jax.numpy as jnp
from flax import nnx


class WNModel(nnx.Module):
  """Weight-normalized linear layer parameterized by direction ``v`` and scale ``g``."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs) -> None:
    """Initialize weight direction, magnitude parameter, and bias.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        rngs: Flax NNX random number generator container.
    """
    self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
    self.weight_g = nnx.Param(jnp.ones((1, out_features)))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Project ``x`` using normalized kernel columns scaled by ``weight_g``.

    Args:
        x: Input JAX array of shape ``(..., in_features)``.

    Returns:
        Weight-normalized linear output array.
    """
    kernel = self.linear.kernel.value
    norm = jnp.linalg.norm(kernel, axis=0, keepdims=True) + 1e-12
    w_normed = kernel * (self.weight_g.value / norm)
    res: jnp.ndarray = jnp.dot(x, w_normed) + self.linear.bias.value
    return res
