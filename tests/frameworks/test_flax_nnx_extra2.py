"""Test module."""

import sys
from unittest.mock import MagicMock, patch


def test_flax_nnx_definitions_no_mock():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = FlaxNNXAdapter()
    d = adapter.definitions
    assert "ReLU" in d
    assert "Linear" in d
    assert "Conv2d" in d


def test_flax_nnx_convert_branch():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class ObjWithArray:
    """An object with __array__."""

    def __array__(self):
      """Gets the array."""
      return [1, 2, 3]

  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj = ObjWithArray()
    assert adapter.convert(obj) is obj


def test_flax_nnx_convert_no_import():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  real_import = __import__

  def mock_import(name, *args, **kwargs):
    """Mocks __import__ to raise ImportError for jax.numpy."""
    if name == "jax.numpy":
      raise ImportError("No module named jax.numpy")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_flax_nnx_convert_not_array_like():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  mock_jnp = MagicMock()
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    assert adapter.convert("not_array_like") == "not_array_like"


def test_flax_nnx_apply_wiring_branches():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot = {
    "mappings": {
      "test1": {"api": "flax.nnx.Test"},
      "test2": {"other": "flax.nnx.Test"},
      "test3": None,
      "forward": {"api": "fwd"},
    }
  }
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["test1"]["api"] == "nnx.Test"
  assert snapshot["mappings"]["__call__"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["call"]["requires_plugin"] == "inject_training_flag"
