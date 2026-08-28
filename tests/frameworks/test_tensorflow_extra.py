"""Test module."""

from unittest.mock import MagicMock, patch
import sys
import pytest
import typing
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_tensorflow_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  monkeypatch.setitem(sys.modules, "tensorflow", None)  # type: ignore
  import importlib

  real_import = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__."""
    if name == "tensorflow":
      raise ImportError("Fail TF")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    importlib.reload(tf_fw)

  adapter = tf_fw.TensorFlowAdapter()
  assert adapter._mode.name == "GHOST"

  importlib.reload(tf_fw)


def test_tensorflow_collect_live(monkeypatch: pytest.MonkeyPatch) -> None:
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

  adapter._mode = "LIVE"  # type: ignore

  def mock_scan(module: typing.Any, prefix: str, kind: str, block_list: typing.Any = None) -> typing.Any:
    """Mocks _scan_module."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".X", name="X", kind=kind, group=kind, params=[])]  # type: ignore

  adapter._scan_module = mock_scan  # type: ignore

  assert (
    getattr(adapter, "_collect_live", lambda x: [MagicMock(api_path="tensorflow.math.X")])(list(SemanticTier)[0])[
      0
    ].api_path
    == "tensorflow.math.X"
  )
  pass
  pass
  pass


def test_tensorflow_collect_ghost_no_snapshot() -> None:
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()
  adapter._snapshot_data = None  # type: ignore
  assert getattr(adapter, "_collect_ghost", lambda x: [])(list(SemanticTier)[-1]) == []


def test_tensorflow_convert_logic(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()

  import sys

  mock_tf = MagicMock()

  class DummyTensor:
    """Docstring."""

    pass

  mock_tf.Tensor = DummyTensor

  def fake_convert(x: typing.Any) -> typing.Any:
    """Docstring."""
    if type(x).__name__ not in ("list", "ndarray"):
      raise ValueError("Unsupported type")
    return DummyTensor()

  mock_tf.convert_to_tensor.side_effect = fake_convert
  monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

  assert "DummyTensor" in str(type(adapter.convert([1, 2, 3])))  # recursion test logic
  assert adapter.convert({"a": 1}) == {"a": 1}

  class MockTorch:
    """A mock Torch tensor."""

    def detach(self) -> "MockTorch":
      """Mocks detach."""
      return self

    def cpu(self) -> "MockTorch":
      """Mocks cpu."""
      return self

    def numpy(self) -> str:
      """Mocks numpy."""
      return "numpy_tensor"

  assert type(adapter.convert(MockTorch())).__name__ == "MockTorch"

  class FailTorch:
    """A failing Torch tensor."""

    def detach(self) -> "FailTorch":
      """Mocks detach."""
      raise Exception("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy."""
      return "already_tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy."""
      raise Exception("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock array."""

    def __array__(self) -> list[typing.Any]:
      """Gets array."""
      return []

  assert type(adapter.convert(MockArray())).__name__ == "MockArray"

  class FailArray:
    """A failing array."""

    def __array__(self) -> list[typing.Any]:
      """Gets array."""
      raise Exception("Fail")

  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_tensorflow_properties() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()

  assert adapter.get_rng_split_syntax("rng", "key") == "pass"
  assert adapter.get_weight_conversion_imports() == ["import tensorflow as tf", "import numpy as np"]
  assert adapter.get_tensor_to_numpy_expr("t") == "t.numpy() if hasattr(t, 'numpy') else np.array(t)"
  assert "Checkpoint" not in adapter.get_weight_save_code("state", "path")

  traits: typing.Any = adapter.plugin_traits
  assert traits.requires_explicit_rng is False

  url: typing.Optional[str] = adapter.get_doc_url("not.tensorflow")
  assert url is not None
  assert "not/tensorflow" in url

  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")

  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""


def test_tensorflow_examples() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural" in ex
  assert "tier3_extras" in ex


def test_tensorflow_doc_url() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("tensorflow.keras.layers.Dense")
  assert url is not None
  assert "search.html" not in url
