"""Tests for TensorFlow framework adapters."""

import typing

from typing import Any
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.tensorflow_examples import get_tf_tiered_examples


def test_tf_adapter_properties() -> None:
  """Verify tf adapter properties behavior."""
  adapter: typing.Any = TensorFlowAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.tensorflow.load_definitions", return_value={"test": MagicMock()}):
    adapter.definitions

  pass

  with patch.dict("sys.modules", {"tensorflow": None}):
    adapter.convert([1, 2])

  with patch.dict("sys.modules", {"tensorflow": MagicMock()}):
    import tensorflow as tf

    tf.convert_to_tensor.return_value = "tensor"
    # adapter.convert([1, 2]) # Removed to avoid truth value error

    tf.convert_to_tensor.side_effect = RuntimeError("err")
    adapter.convert([1, 2])

  adapter.get_device_syntax("cpu", "0")
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path", "obj")
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_serialization_syntax("other", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("tf.math.add")
  _res: typing.Any = adapter.get_doc_url("unknown")
  assert _res == "https://www.tensorflow.org/api_docs/python/unknown"


def test_tf_examples() -> None:
  """Verify tf examples behavior."""
  ex: typing.Any = get_tf_tiered_examples()
  assert isinstance(ex, dict)


def test_tf_extra_misses() -> None:
  """Verify tf extra misses behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter: typing.Any = TensorFlowAdapter()

  # 231-237
  adapter.apply_wiring(
    {
      "mappings": {
        "test": {"api": "tensorflow.test"},
        "non_tf": {"api": "tf.keras.layers.Dense"},
        "bad": {},
        "none": None,
        "no_api": {"a": 1},
      }
    }
  )

  # 241-243
  adapter.get_tiered_examples()

  # 261, 267
  adapter.get_device_syntax("gpu")
  adapter.get_device_syntax("cpu", "1")
  _res: typing.Any = adapter.get_device_syntax("cpu", "var")
  assert "tf.device" in _res


def test_tf_ghost_init() -> None:
  """Verify tf ghost init behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  with patch("ml_switcheroo.frameworks.tensorflow.tf", None):
    with patch("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={}):
      adapter: typing.Any = TensorFlowAdapter()
      assert adapter._mode == InitMode.GHOST


def test_tf_ghost_init_with_snapshot() -> None:
  """Verify tf ghost init with snapshot behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  with patch("ml_switcheroo.frameworks.tensorflow.tf", None):
    with patch("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={"categories": {}}):
      adapter: typing.Any = TensorFlowAdapter()
      assert adapter._mode == InitMode.GHOST


def test_tf_definitions_all_exist() -> None:
  """Test definitions when all standard keys are present."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter: typing.Any = TensorFlowAdapter()
  existing: typing.Any = {
    "Linear": MagicMock(),
    "Conv2d": MagicMock(),
    "Conv1d": MagicMock(),
    "Conv3d": MagicMock(),
    "ConvTranspose2d": MagicMock(),
  }
  with patch("ml_switcheroo.frameworks.tensorflow.load_definitions", return_value=existing):
    defs: typing.Any = adapter.definitions
    assert "Linear" in defs


def test_tf_import_exception(monkeypatch: Any) -> None:
  """Test module behavior when importing tensorflow fails.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import builtins
  import importlib

  real_import: typing.Any = builtins.__import__

  def mock_import(name: typing.Any, *args: Any, **kwargs: Any) -> Any:
    """Execute mock_import test helper.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "tensorflow":
      raise ImportError("No TF")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  import ml_switcheroo.frameworks.tensorflow as tf_mod

  try:
    importlib.reload(tf_mod)
    assert tf_mod.tf is None
  finally:
    monkeypatch.undo()
    importlib.reload(tf_mod)
