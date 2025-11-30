import jax
import jax.numpy as jnp


def parallel_step(x):
  # This standard op SHOULD be converted to Torch
  val = jnp.abs(x)

  # This parallelization primitive should trigger the Escape Hatch
  # as it cannot be trivially mapped to a torch function call.
  out = jax.pmap(lambda v: v * 2)(val)

  return out
