"""Module docstring."""

from flax import nnx
import jax.numpy as jnp
import jax


def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray, loss_fn) -> jnp.ndarray:
  """Function docstring."""

  # <SWITCHEROO_FAILED_TO_TRANS>
  # Functional stateless gradients using value_and_grad
  def loss_closure(model_ref: nnx.Module):
    """Function docstring."""
    predictions = model_ref(x)
    return loss_fn(predictions, y)

  loss, grads = nnx.value_and_grad(loss_closure)(model)
  optimizer.update(grads)
  return loss
  # </SWITCHEROO_FAILED_TO_TRANS>
