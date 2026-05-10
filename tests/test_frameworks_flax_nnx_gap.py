import types
from unittest import mock
from ml_switcheroo.frameworks.base import StandardCategory, InitMode


def test_flax_nnx_import_success():
  import ml_switcheroo.frameworks.flax_nnx as fnnx
  import importlib

  mock_flax = mock.MagicMock()
  mock_flax_nnx = mock.MagicMock()
  mock_flax.nnx = mock_flax_nnx

  with mock.patch.dict("sys.modules", {"flax": mock_flax, "flax.nnx": mock_flax_nnx}):
    importlib.reload(fnnx)
    assert fnnx.flax_nnx is not None


def test_flax_nnx_collect_api():
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  adapter._mode = InitMode.LIVE
  adapter._flax_available = True

  refs = adapter.collect_api(StandardCategory.LAYER)
  assert isinstance(refs, list)


def test_flax_nnx_collect_api_with_mocked_flax(caplog):
  import logging

  caplog.set_level(logging.DEBUG)

  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  class MockModule:
    pass

  class ValidModule(MockModule):
    pass

  mock_nnx_mod = types.ModuleType("flax.nnx")
  mock_nnx_mod.Module = MockModule
  mock_nnx_mod.Valid = ValidModule
  mock_nnx_mod._hidden = ValidModule
  mock_nnx_mod.exploder = "exploder"

  mock_flax_mod = types.ModuleType("flax")
  mock_flax_mod.nnx = mock_nnx_mod
  mock_flax_mod.__path__ = []

  import inspect

  orig_isclass = inspect.isclass

  def mock_isclass(obj):
    if obj == "exploder":
      raise Exception("boom")
    return orig_isclass(obj)

  adapter = FlaxNNXAdapter()
  adapter._mode = InitMode.LIVE
  adapter._flax_available = True

  with mock.patch.dict("sys.modules", {"flax": mock_flax_mod, "flax.nnx": mock_nnx_mod}):
    import ml_switcheroo.frameworks.flax_nnx

    ml_switcheroo.frameworks.flax_nnx.flax_nnx = mock_nnx_mod
    with mock.patch("inspect.isclass", mock_isclass):
      with mock.patch("ml_switcheroo.core.ghost.GhostInspector.inspect", return_value="ghost"):
        refs = adapter.collect_api(StandardCategory.LAYER)
        assert len(refs) > 0


def test_flax_nnx_convert_array():
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  class ExploderList(list):
    pass

  e = ExploderList(["explode"])

  def mock_array(data):
    if data is e:
      raise Exception("boom")
    return ["array", data]

  class MockJNP:
    array = staticmethod(mock_array)

  mock_jnp = MockJNP()
  mock_jax = mock.MagicMock()
  mock_jax.numpy = mock_jnp

  with mock.patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jnp}):
    adapter = FlaxNNXAdapter()

    res = adapter.convert([1, 2, 3])
    assert res == ["array", [1, 2, 3]]

    res2 = adapter.convert(e)
    assert res2 is e
