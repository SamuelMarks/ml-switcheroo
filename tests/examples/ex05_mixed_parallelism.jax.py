"""Example illustrating mixed parallelism in JAX.

This module provides a basic example of applying JAX operations that combine
standard elementwise mathematical functions (like jnp.abs) with parallel mapping
operators (like jax.pmap). In the context of ml-switcheroo, this example is used
to verify that while standard ops can be transpiled safely, the presence of pmap triggers
the escape hatch translation mechanism.
"""

import jax
import jax.numpy as jnp


def parallel_step(x):
  """Performs a parallel computation step using JAX.

  First computes the absolute value of the input array elements, then applies
  a parallel map (pmap) operation to double each element along the mapped axis.

  Args:
    x: Input array (jax.Array or jnp.ndarray) containing numeric elements.

  Returns:
    jax.Array: The result of applying the parallel doubling operation to the
      absolute values of the input array.
  """
  # This standard op SHOULD be converted to Torch
  val = jnp.abs(x)

  # This parallelization primitive should trigger the Escape Hatch
  # as it cannot be trivially mapped to a torch function call.
  out = jax.pmap(lambda v: v * 2)(val)

  return out
