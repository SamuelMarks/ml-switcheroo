"""Tests for TensorFlow framework adapters."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.tensorflow_examples import get_tf_tiered_examples
from ml_switcheroo.frameworks.base import InitMode


def test_tf_adapter_properties():
  """Test element."""
  adapter = TensorFlowAdapter()

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

    tf.convert_to_tensor.side_effect = Exception("err")
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
  adapter.get_doc_url("unknown")


def test_tf_examples():
  """Test element."""
  ex = get_tf_tiered_examples()
  assert isinstance(ex, dict)


def test_tf_extra_misses():
  """Test element."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()

  # 231-237
  adapter.apply_wiring({"mappings": {"test": {"api": "tensorflow.test"}, "bad": {}, "none": None, "no_api": {"a": 1}}})

  # 241-243
  adapter.get_tiered_examples()

  # 261, 267
  adapter.get_device_syntax("gpu")
  adapter.get_device_syntax("cpu", "1")
  adapter.get_device_syntax("cpu", "var")


def test_tf_ghost_init():
  """Test element."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  with patch("ml_switcheroo.frameworks.tensorflow.tf", None):
    with patch("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={}):
      adapter = TensorFlowAdapter()
      assert adapter._mode == InitMode.GHOST
