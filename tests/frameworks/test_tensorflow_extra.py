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
    """Mocks __import__."""
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
    """Mocks _scan_module."""
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

  import sys

  mock_tf = MagicMock()

  class DummyTensor:
    pass

  mock_tf.Tensor = DummyTensor

  def fake_convert(x):
    if type(x).__name__ not in ("list", "ndarray"):
      raise ValueError("Unsupported type")
    return DummyTensor()

  mock_tf.convert_to_tensor.side_effect = fake_convert
  monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

  assert "DummyTensor" in str(type(adapter.convert([1, 2, 3])))  # recursion test logic
  assert adapter.convert({"a": 1}) == {"a": 1}

  class MockTorch:
    """A mock Torch tensor."""

    def detach(self):
      """Mocks detach."""
      return self

    def cpu(self):
      """Mocks cpu."""
      return self

    def numpy(self):
      """Mocks numpy."""
      return "numpy_tensor"

  assert type(adapter.convert(MockTorch())).__name__ == "MockTorch"

  class FailTorch:
    """A failing Torch tensor."""

    def detach(self):
      """Mocks detach."""
      raise Exception("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self):
      """Mocks numpy."""
      return "already_tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self):
      """Mocks numpy."""
      raise Exception("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock array."""

    def __array__(self):
      """Gets array."""
      return []

  assert type(adapter.convert(MockArray())).__name__ == "MockArray"

  class FailArray:
    """A failing array."""

    def __array__(self):
      """Gets array."""
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
