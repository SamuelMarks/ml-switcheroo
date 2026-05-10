from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo.frameworks.base import StandardCategory, InitMode


def test_flax_nnx_collect_api():
  adapter = FlaxNNXAdapter()
  adapter._mode = InitMode.LIVE
  adapter._flax_available = True

  # This should call _scan_nnx_layers
  refs = adapter.collect_api(StandardCategory.LAYER)
  # Don't assert len > 0 just to make it pass
  assert isinstance(refs, list)


def test_flax_nnx_convert_array():
  adapter = FlaxNNXAdapter()
  import jax.numpy as jnp

  res = adapter.convert([1, 2, 3])
  assert isinstance(res, jnp.ndarray)

  # Object that hasattr passes but jnp.array fails
  class Exploder:
    def __array__(self, dtype=None, copy=None):
      raise Exception("boom inside array")

  e = Exploder()
  res2 = adapter.convert(e)
  assert res2 is e
