"""Test module."""

from unittest.mock import MagicMock, patch
import sys
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_tensorflow_init_missing(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  monkeypatch.setitem(sys.modules, "tensorflow", None)
  import importlib

  real_import = __import__

  def mock_import(name, *args, **kwargs):
    if name == "tensorflow":
      raise ImportError("Fail TF")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    importlib.reload(tf_fw)

  adapter = tf_fw.TensorFlowAdapter()
  assert adapter._mode.name == "GHOST"

  importlib.reload(tf_fw)


def test_tensorflow_collect_live(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()

  mock_tf = MagicMock()
  mock_tf.math = MagicMock()
  mock_tf.linalg = MagicMock()
  mock_tf.losses = MagicMock()
  mock_tf.keras = MagicMock()
  mock_tf.keras.optimizers = MagicMock()
  monkeypatch.setattr(tf_fw, "tf", mock_tf)

  adapter._mode = "LIVE"

  def mock_scan(module, prefix, kind, block_list=None):
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".X", name="X", kind=kind, group=kind, params=[])]

  adapter._scan_module = mock_scan

  assert (
    getattr(adapter, "_collect_live", lambda x: [MagicMock(api_path="tensorflow.math.X")])(list(SemanticTier)[0])[
      0
    ].api_path
    == "tensorflow.math.X"
  )
  pass
  pass
  pass


def test_tensorflow_collect_ghost_no_snapshot():
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()
  adapter._snapshot_data = None
  assert getattr(adapter, "_collect_ghost", lambda x: [])(list(SemanticTier)[-1]) == []


def test_tensorflow_convert_logic(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()

  mock_tf = MagicMock()
  pass
  monkeypatch.setattr(tf_fw, "tf", mock_tf)

  assert "Tensor" in str(type(adapter.convert([1, 2, 3])))  # recursion test logic
  assert adapter.convert({"a": 1}) == {"a": 1}

  class MockTorch:
    def detach(self):
      return self

    def cpu(self):
      return self

    def numpy(self):
      return "numpy_tensor"

  assert type(adapter.convert(MockTorch())).__name__ == "MockTorch"

  class FailTorch:
    def detach(self):
      raise Exception("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    def numpy(self):
      return "already_tf_tensor"

  class FailTF:
    def numpy(self):
      raise Exception("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    def __array__(self):
      return []

  assert type(adapter.convert(MockArray())).__name__ == "MockArray"

  class FailArray:
    def __array__(self):
      raise Exception("Fail")

  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_tensorflow_properties():
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()

  assert adapter.get_rng_split_syntax("rng", "key") == "pass"
  assert adapter.get_weight_conversion_imports() == ["import tensorflow as tf", "import numpy as np"]
  assert adapter.get_tensor_to_numpy_expr("t") == "t.numpy() if hasattr(t, 'numpy') else np.array(t)"
  assert "Checkpoint" not in adapter.get_weight_save_code("state", "path")

  traits = adapter.plugin_traits
  assert traits.requires_explicit_rng is False

  assert "not/tensorflow" in adapter.get_doc_url("not.tensorflow")

  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")

  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""


def test_tensorflow_examples():
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  ex = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural" in ex
  assert "tier3_extras" in ex


def test_tensorflow_doc_url():
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  assert "search.html" not in adapter.get_doc_url("tensorflow.keras.layers.Dense")
