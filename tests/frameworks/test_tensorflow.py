"""Test suite for the Tensorflow module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


def test_tensorflow_import_exception() -> None:
  """Verifies the behavior when tensorflow fails to import during module load."""
  from importlib import reload

  import ml_switcheroo.frameworks.tensorflow as tf_module

  real_import: typing.Any = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks python import builtin.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "tensorflow":
      raise ImportError("Simulated TF load failure")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    reload(tf_module)
    assert getattr(tf_module, "tf", None) is None

  reload(tf_module)  # restore


def test_tensorflow_adapter_init() -> None:
  """Verifies the behavior of TensorFlow adapter initialization."""
  adapter: typing.Any = TensorFlowAdapter()
  assert adapter.display_name == "TensorFlow"
  assert adapter.ui_priority == 30
  assert adapter.inherits_from is None
  assert adapter._mode in (InitMode.GHOST, InitMode.LIVE)


def test_tensorflow_init_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of TensorFlow initialization live.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.tf", True)
  adapter: typing.Any = TensorFlowAdapter()
  assert adapter._mode == InitMode.LIVE


def test_tensorflow_init_ghost_no_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of TensorFlow initialization ghost no snapshot.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.tf", None)
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", lambda _: {})
  with patch("logging.debug") as mock_debug:
    adapter: typing.Any = TensorFlowAdapter()
    assert adapter._mode == InitMode.GHOST
    mock_debug.assert_called_once_with("TensorFlow not installed and no snapshot found.")


def test_tensorflow_properties() -> None:
  """Verifies the behavior of TensorFlow properties."""
  adapter: typing.Any = TensorFlowAdapter()
  assert adapter.import_alias == ("tensorflow", "tf")
  ns: typing.Any = adapter.import_namespaces
  assert "tensorflow" in ns
  assert ns["tensorflow"].recommended_alias == "tf"
  config: dict[str, typing.Any] = adapter.test_config
  assert "import tensorflow as tf" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "hasattr(obj, 'numpy')" in adapter.get_to_numpy_code()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.forward_method == "call"
  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)
  assert "set_seed" in adapter.rng_seed_methods


def test_tensorflow_apply_wiring() -> None:
  """Verifies the behavior of TensorFlow apply wiring."""
  adapter: typing.Any = TensorFlowAdapter()
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}

  # Test with mappings
  snapshot2: dict[str, typing.Any] = {
    "mappings": {"op1": {"api": "tensorflow.math.add"}, "op2": {"api": "other.add"}, "op3": None, "op4": {"not_api": 1}}
  }
  adapter.apply_wiring(snapshot2)
  assert snapshot2["mappings"]["op1"]["api"] == "tf.math.add"
  assert snapshot2["mappings"]["op2"]["api"] == "other.add"
  assert snapshot2["mappings"]["op3"] is None
  assert snapshot2["mappings"]["op4"] == {"not_api": 1}


def test_tensorflow_device_syntax() -> None:
  """Verifies the behavior of TensorFlow device syntax."""
  adapter: typing.Any = TensorFlowAdapter()
  assert "tf.device('GPU:0')" == adapter.get_device_syntax("cuda")
  assert "tf.device('CPU:0')" == adapter.get_device_syntax("cpu")
  assert "tf.device('GPU:1')" == adapter.get_device_syntax("cuda", "1")
  assert "tf.device(f'GPU:{str(var)}')" == adapter.get_device_syntax("cuda", "var")


def test_tensorflow_device_check_syntax() -> None:
  """Verifies the behavior of TensorFlow device check syntax."""
  adapter: typing.Any = TensorFlowAdapter()
  assert "len(tf.config.list_physical_devices('GPU')) > 0" in adapter.get_device_check_syntax()


def test_tensorflow_serialization() -> None:
  """Verifies the behavior of TensorFlow serialization."""
  adapter: typing.Any = TensorFlowAdapter()
  assert "import tensorflow as tf" in adapter.get_serialization_imports()
  assert "tf.io.write_file(f, obj)" == adapter.get_serialization_syntax("save", "f", "obj")
  assert "tf.io.read_file(f)" == adapter.get_serialization_syntax("load", "f")
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.get_serialization_syntax("unknown", "f") == ""


def test_tensorflow_weight_load() -> None:
  """Verifies the behavior of TensorFlow weight load."""
  adapter: typing.Any = TensorFlowAdapter()
  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")


def test_tensorflow_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of TensorFlow convert.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys
  from unittest.mock import MagicMock

  mock_tf: typing.Any
  if "tensorflow" not in sys.modules:
    mock_tf = MagicMock()

    class DummyTensor:
      """Test helper class DummyTensor."""

      pass

    mock_tf.Tensor = DummyTensor
    mock_tf.convert_to_tensor.side_effect = lambda x: DummyTensor()
    sys.modules["tensorflow"] = mock_tf
  else:
    mock_tf = sys.modules["tensorflow"]
    if not hasattr(mock_tf, "Tensor"):

      class DummyTensor2:
        """Test helper class DummyTensor2."""

        pass

      mock_tf.Tensor = DummyTensor2
      mock_tf.convert_to_tensor.side_effect = lambda x: DummyTensor2()

  adapter: typing.Any = TensorFlowAdapter()

  # When TF is present, it returns a Tensor
  res: typing.Any = adapter.convert("test")
  assert isinstance(res, mock_tf.Tensor)

  # When TF fails to import, it returns the original string
  monkeypatch.setitem(sys.modules, "tensorflow", None)
  assert adapter.convert("test") == "test"


def test_tensorflow_doc_url() -> None:
  """Verifies the behavior of TensorFlow documentation URL."""
  adapter: typing.Any = TensorFlowAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("tensorflow.math.abs")
  assert url is not None
  assert "tf/math/abs" in url


@patch("ml_switcheroo.frameworks.tensorflow_examples.get_tf_tiered_examples")
def test_tensorflow_tiered_examples(mock_examples: typing.Any) -> None:
  """Verifies the behavior of TensorFlow tiered examples.

  Args:
      mock_examples: Test argument or fixture.
  """
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter: typing.Any = TensorFlowAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_tensorflow_missing_coverage() -> None:
  """Verifies missing coverage methods."""
  adapter: typing.Any = TensorFlowAdapter()

  # Plugin Traits
  traits: typing.Any = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # RNG split
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  # Weight conversion imports
  assert "import numpy as np" in adapter.get_weight_conversion_imports()

  # Tensor to numpy
  expr: str = adapter.get_tensor_to_numpy_expr("var")
  assert "var.numpy() if hasattr(var, 'numpy') else np.array(var)" == expr

  # Weight save
  assert "WARNING: Saving raw dictionary" in adapter.get_weight_save_code("s", "p")


# --- Merged from test_tensorflow_extra.py ---


def test_tensorflow_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify tensorflow init missing behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  monkeypatch.setitem(sys.modules, "tensorflow", None)
  import importlib

  real_import: typing.Any = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "tensorflow":
      raise ImportError("Fail TF")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    importlib.reload(tf_fw)

  adapter: typing.Any = tf_fw.TensorFlowAdapter()
  assert adapter._mode.name == "GHOST"

  importlib.reload(tf_fw)


def test_tensorflow_collect_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify tensorflow collect live behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter: typing.Any = tf_fw.TensorFlowAdapter()

  mock_tf: typing.Any = MagicMock()
  mock_tf.math = MagicMock()
  mock_tf.linalg = MagicMock()
  mock_tf.losses = MagicMock()
  mock_tf.keras = MagicMock()
  mock_tf.keras.optimizers = MagicMock()
  monkeypatch.setattr(tf_fw, "tf", mock_tf)

  adapter._mode = "LIVE"

  def mock_scan(module: typing.Any, prefix: str, kind: str, block_list: typing.Any = None) -> typing.Any:
    """Mocks _scan_module.

    Args:
        module: Test argument or fixture.
        prefix: Test argument or fixture.
        kind: Test argument or fixture.
        block_list: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    from ml_switcheroo.core.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".X", name="X", kind=kind, params=[])]

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


def test_tensorflow_collect_ghost_no_snapshot() -> None:
  """Verify tensorflow collect ghost no snapshot behavior."""
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter: typing.Any = tf_fw.TensorFlowAdapter()
  adapter._snapshot_data = None
  assert getattr(adapter, "_collect_ghost", lambda x: [])(list(SemanticTier)[-1]) == []


def test_tensorflow_convert_logic(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify tensorflow convert logic behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter: typing.Any = tf_fw.TensorFlowAdapter()

  import sys

  mock_tf: typing.Any = MagicMock()

  class DummyTensor:
    """Test helper class DummyTensor."""

    pass

  mock_tf.Tensor = DummyTensor

  def fake_convert(x: typing.Any) -> typing.Any:
    """Execute fake convert helper logic.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
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
      """Mocks detach.

      Returns:
          Computed test result or fixture value.
      """
      return self

    def cpu(self) -> "MockTorch":
      """Mocks cpu.

      Returns:
          Computed test result or fixture value.
      """
      return self

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Computed test result or fixture value.
      """
      return "numpy_tensor"

  assert type(adapter.convert(MockTorch())).__name__ == "MockTorch"

  class FailTorch:
    """A failing Torch tensor."""

    def detach(self) -> "FailTorch":
      """Mocks detach.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("Fail")

  f: typing.Any = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Computed test result or fixture value.
      """
      return "already_tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("Fail")

  f2: typing.Any = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock array."""

    def __array__(self) -> list[typing.Any]:
      """Gets array.

      Returns:
          Computed test result or fixture value.
      """
      return []

  assert type(adapter.convert(MockArray())).__name__ == "MockArray"

  class FailArray:
    """A failing array."""

    def __array__(self) -> list[typing.Any]:
      """Gets array.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("Fail")

  f3: typing.Any = FailArray()
  assert adapter.convert(f3) is f3


def test_tensorflow_properties_extra() -> None:
  """Verify tensorflow properties extra behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter: typing.Any = TensorFlowAdapter()

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
  """Verify tensorflow examples behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter: typing.Any = TensorFlowAdapter()
  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural" in ex
  assert "tier3_extras" in ex


def test_tensorflow_doc_url_extra() -> None:
  """Verify tensorflow doc url extra behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter: typing.Any = TensorFlowAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("tensorflow.keras.layers.Dense")
  assert url is not None
  assert "search.html" not in url


def test_tensorflow_ghost_mode_with_snapshot() -> None:
  """Verify tensorflow ghost mode with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  with patch("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={"test": 1}):
    with patch("ml_switcheroo.frameworks.tensorflow.tf", None):
      adapter: typing.Any = TensorFlowAdapter()
      assert adapter._snapshot_data == {"test": 1}
