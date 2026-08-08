"""Test suite for the Tensorflow module."""

from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier
from unittest.mock import patch


def test_tensorflow_import_exception():
  """Verifies the behavior when tensorflow fails to import during module load."""
  from importlib import reload
  import ml_switcheroo.frameworks.tensorflow as tf_module

  real_import = __import__

  def mock_import(name, *args, **kwargs):
    """Mocks python import builtin."""
    if name == "tensorflow":
      raise Exception("Simulated TF load failure")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    reload(tf_module)
    assert tf_module.tf is None

  reload(tf_module)  # restore


def test_tensorflow_adapter_init():
  """Verifies the behavior of TensorFlow adapter initialization."""
  adapter = TensorFlowAdapter()
  assert adapter.display_name == "TensorFlow"
  assert adapter.ui_priority == 30
  assert adapter.inherits_from is None
  assert adapter._mode in (InitMode.GHOST, InitMode.LIVE)


def test_tensorflow_init_live(monkeypatch):
  """Verifies the behavior of TensorFlow initialization live."""
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.tf", True)
  adapter = TensorFlowAdapter()
  assert adapter._mode == InitMode.LIVE


def test_tensorflow_init_ghost_no_snapshot(monkeypatch):
  """Verifies the behavior of TensorFlow initialization ghost no snapshot."""
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.tf", None)
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", lambda _: {})
  with patch("logging.debug") as mock_debug:
    adapter = TensorFlowAdapter()
    assert adapter._mode == InitMode.GHOST
    mock_debug.assert_called_once_with("TensorFlow not installed and no snapshot found.")


def test_tensorflow_properties():
  """Verifies the behavior of TensorFlow properties."""
  adapter = TensorFlowAdapter()
  assert adapter.import_alias == ("tensorflow", "tf")
  ns = adapter.import_namespaces
  assert "tensorflow" in ns
  assert ns["tensorflow"].recommended_alias == "tf"
  config = adapter.test_config
  assert "import tensorflow as tf" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "hasattr(obj, 'numpy')" in adapter.get_to_numpy_code()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  traits = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.forward_method == "call"
  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert "set_seed" in adapter.rng_seed_methods


def test_tensorflow_apply_wiring():
  """Verifies the behavior of TensorFlow apply wiring."""
  adapter = TensorFlowAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}

  # Test with mappings
  snapshot2 = {
    "mappings": {"op1": {"api": "tensorflow.math.add"}, "op2": {"api": "other.add"}, "op3": None, "op4": {"not_api": 1}}
  }
  adapter.apply_wiring(snapshot2)
  assert snapshot2["mappings"]["op1"]["api"] == "tf.math.add"
  assert snapshot2["mappings"]["op2"]["api"] == "other.add"
  assert snapshot2["mappings"]["op3"] is None
  assert snapshot2["mappings"]["op4"] == {"not_api": 1}


def test_tensorflow_device_syntax():
  """Verifies the behavior of TensorFlow device syntax."""
  adapter = TensorFlowAdapter()
  assert "tf.device('GPU:0')" == adapter.get_device_syntax("cuda")
  assert "tf.device('CPU:0')" == adapter.get_device_syntax("cpu")
  assert "tf.device('GPU:1')" == adapter.get_device_syntax("cuda", "1")
  assert "tf.device(f'GPU:{str(var)}')" == adapter.get_device_syntax("cuda", "var")


def test_tensorflow_device_check_syntax():
  """Verifies the behavior of TensorFlow device check syntax."""
  adapter = TensorFlowAdapter()
  assert "len(tf.config.list_physical_devices('GPU')) > 0" in adapter.get_device_check_syntax()


def test_tensorflow_serialization():
  """Verifies the behavior of TensorFlow serialization."""
  adapter = TensorFlowAdapter()
  assert "import tensorflow as tf" in adapter.get_serialization_imports()
  assert "tf.io.write_file(f, obj)" == adapter.get_serialization_syntax("save", "f", "obj")
  assert "tf.io.read_file(f)" == adapter.get_serialization_syntax("load", "f")
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.get_serialization_syntax("unknown", "f") == ""


def test_tensorflow_weight_load():
  """Verifies the behavior of TensorFlow weight load."""
  adapter = TensorFlowAdapter()
  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")


def test_tensorflow_convert(monkeypatch):
  """Verifies the behavior of TensorFlow convert."""
  import tensorflow as tf
  import sys

  adapter = TensorFlowAdapter()

  # When TF is present, it returns a Tensor
  res = adapter.convert("test")
  assert isinstance(res, tf.Tensor)

  # When TF fails to import, it returns the original string
  monkeypatch.setitem(sys.modules, "tensorflow", None)
  assert adapter.convert("test") == "test"


def test_tensorflow_doc_url():
  """Verifies the behavior of TensorFlow documentation URL."""
  adapter = TensorFlowAdapter()
  url = adapter.get_doc_url("tensorflow.math.abs")
  assert "tf/math/abs" in url


@patch("ml_switcheroo.frameworks.tensorflow_examples.get_tf_tiered_examples")
def test_tensorflow_tiered_examples(mock_examples):
  """Verifies the behavior of TensorFlow tiered examples."""
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = TensorFlowAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_tensorflow_missing_coverage():
  """Verifies missing coverage methods."""
  adapter = TensorFlowAdapter()

  # Plugin Traits
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # RNG split
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  # Weight conversion imports
  assert "import numpy as np" in adapter.get_weight_conversion_imports()

  # Tensor to numpy
  expr = adapter.get_tensor_to_numpy_expr("var")
  assert "var.numpy() if hasattr(var, 'numpy') else np.array(var)" == expr

  # Weight save
  assert "WARNING: Saving raw dictionary" in adapter.get_weight_save_code("s", "p")
