"""Type stubs for ``praxis.optimizers``."""

import jax.numpy as jnp

class _AdamHParams:
  """Typed stub for ``praxis.optimizers.Adam.HParams``."""

  def __call__(self, *, learning_rate: float = ...) -> jnp.ndarray: ...

class Adam:
  """Typed stub for ``praxis.optimizers.Adam``."""

  HParams: _AdamHParams
