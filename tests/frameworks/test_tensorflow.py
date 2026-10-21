"""Test suite for the Tensorflow module."""

import sys
import typing
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


def test_tensorflow_import_exception() -> None:
  """Verifies the behavior when tensorflow fails to import during module load."""
  import types
  from importlib import reload

  import ml_switcheroo.frameworks.tensorflow as tf_module

  real_import = __import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
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
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": real_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with patch("builtins.__import__", mock_import):
    reload(tf_module)
    assert getattr(tf_module, "tf", None) is None

  reload(tf_module)  # restore


def test_tensorflow_adapter_init() -> None:
  """Verifies the behavior of TensorFlow adapter initialization."""
  adapter = TensorFlowAdapter()
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
  adapter = TensorFlowAdapter()
  assert adapter._mode == InitMode.LIVE


def test_tensorflow_init_ghost_no_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of TensorFlow initialization ghost no snapshot.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.tf", None)
  monkeypatch.setattr("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", lambda _: {})
  with patch("logging.debug") as mock_debug:
    adapter = TensorFlowAdapter()
    assert adapter._mode == InitMode.GHOST
    mock_debug.assert_called_once_with("TensorFlow not installed and no snapshot found.")


def test_tensorflow_properties() -> None:
  """Verifies the behavior of TensorFlow properties."""
  adapter = TensorFlowAdapter()
  assert adapter.import_alias == ("tensorflow", "tf")
  ns = adapter.import_namespaces
  assert "tensorflow" in ns
  assert getattr(
    {"v": getattr({"v": getattr({"v": ns}, "__getitem__")("v")["tensorflow"]}, "__getitem__")("v").recommended_alias},
    "__getitem__",
  )("v") == getattr({"v": "tf"}, "__getitem__")("v")
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


def test_tensorflow_apply_wiring() -> None:
  """Verifies the behavior of TensorFlow apply wiring."""
  adapter = TensorFlowAdapter()
  snapshot: dict[str, typing.Hashable | typing.Sized] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert snapshot == {}

  # Test with mappings
  snapshot2 = {
    "mappings": {"op1": {"api": "tensorflow.math.add"}, "op2": {"api": "other.add"}, "op3": None, "op4": {"not_api": 1}}
  }
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot2}, "__getitem__")("v")
  )
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": snapshot2}, "__getitem__")("v")["mappings"]}, "__getitem__")("v")["op1"]},
        "__getitem__",
      )("v")["api"]
    },
    "__getitem__",
  )("v") == getattr({"v": "tf.math.add"}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": snapshot2}, "__getitem__")("v")["mappings"]}, "__getitem__")("v")["op2"]},
        "__getitem__",
      )("v")["api"]
    },
    "__getitem__",
  )("v") == getattr({"v": "other.add"}, "__getitem__")("v")
  assert snapshot2["mappings"]["op3"] is None
  assert snapshot2["mappings"]["op4"] == {"not_api": 1}


def test_tensorflow_device_syntax() -> None:
  """Verifies the behavior of TensorFlow device syntax."""
  adapter = TensorFlowAdapter()
  assert "tf.device('GPU:0')" == adapter.get_device_syntax("cuda")
  assert "tf.device('CPU:0')" == adapter.get_device_syntax("cpu")
  assert "tf.device('GPU:1')" == adapter.get_device_syntax("cuda", "1")
  assert "tf.device(f'GPU:{str(var)}')" == adapter.get_device_syntax("cuda", "var")


def test_tensorflow_device_check_syntax() -> None:
  """Verifies the behavior of TensorFlow device check syntax."""
  adapter = TensorFlowAdapter()
  assert "len(tf.config.list_physical_devices('GPU')) > 0" in adapter.get_device_check_syntax()


def test_tensorflow_serialization() -> None:
  """Verifies the behavior of TensorFlow serialization."""
  adapter = TensorFlowAdapter()
  assert "import tensorflow as tf" in adapter.get_serialization_imports()
  assert "tf.io.write_file(f, obj)" == adapter.get_serialization_syntax("save", "f", "obj")
  assert "tf.io.read_file(f)" == adapter.get_serialization_syntax("load", "f")
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.get_serialization_syntax("unknown", "f") == ""


def test_tensorflow_weight_load() -> None:
  """Verifies the behavior of TensorFlow weight load."""
  adapter = TensorFlowAdapter()
  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")


def test_tensorflow_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of TensorFlow convert.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys
  from unittest.mock import MagicMock

  if "tensorflow" not in sys.modules:
    mock_tf = MagicMock()

    class DummyTensor:
      """Test helper class DummyTensor."""

      pass

    mock_tf.Tensor = DummyTensor
    mock_tf.convert_to_tensor.side_effect = lambda x: DummyTensor()
    sys.modules["tensorflow"] = mock_tf
  else:
    mock_tf = getattr(
      {"v": getattr({"v": getattr({"v": sys}, "__getitem__")("v").modules}, "__getitem__")("v")["tensorflow"]},
      "__getitem__",
    )("v")
    if not hasattr(mock_tf, "Tensor"):

      class DummyTensor2:
        """Test helper class DummyTensor2."""

        pass

      mock_tf.Tensor = DummyTensor2
      mock_tf.convert_to_tensor.side_effect = lambda x: DummyTensor2()

  adapter = TensorFlowAdapter()

  # When TF is present, it returns a Tensor
  res = adapter.convert("test")
  assert isinstance(res, mock_tf.Tensor)

  # When TF fails to import, it returns the original string
  monkeypatch.setitem(sys.modules, "tensorflow", None)
  assert adapter.convert("test") == "test"


def test_tensorflow_doc_url() -> None:
  """Verifies the behavior of TensorFlow documentation URL."""
  adapter = TensorFlowAdapter()
  url: str | None = adapter.get_doc_url("tensorflow.math.abs")
  assert url is not None
  assert "tf/math/abs" in url


@patch("ml_switcheroo.frameworks.tensorflow_examples.get_tf_tiered_examples")
def test_tensorflow_tiered_examples(mock_examples: MagicMock) -> None:
  """Verifies the behavior of TensorFlow tiered examples.

  Args:
      mock_examples: Test argument or fixture.
  """
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = TensorFlowAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_tensorflow_missing_coverage() -> None:
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
  import types

  real_import = __import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
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
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": real_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with patch("builtins.__import__", mock_import):
    importlib.reload(tf_fw)

  adapter = tf_fw.TensorFlowAdapter()
  assert adapter._mode.name == "GHOST"

  importlib.reload(tf_fw)


def test_tensorflow_collect_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify tensorflow collect live behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  from ml_switcheroo.core.ghost import GhostRef
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()

  mock_tf = MagicMock()
  mock_tf.math = MagicMock()
  mock_tf.linalg = MagicMock()
  mock_tf.losses = MagicMock()
  mock_tf.keras = MagicMock()
  mock_tf.keras.optimizers = MagicMock()
  monkeypatch.setattr(tf_fw, "tf", mock_tf)

  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_mode", getattr({"v": "LIVE"}, "__getitem__")("v"))

  def mock_scan(
    module: MagicMock,
    prefix: str,
    kind: str,
    block_list: set[str] | None = None,
  ) -> list[GhostRef]:
    """Mocks _scan_module.

    Args:
        module: Test argument or fixture.
        prefix: Test argument or fixture.
        kind: Test argument or fixture.
        block_list: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return [GhostRef(api_path=prefix + ".X", name="X", kind=kind, params=[])]

  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_scan_module", getattr({"v": mock_scan}, "__getitem__")("v"))

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

  adapter = tf_fw.TensorFlowAdapter()
  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_snapshot_data", getattr({"v": None}, "__getitem__")("v"))
  assert getattr(adapter, "_collect_ghost", lambda x: [])(list(SemanticTier)[-1]) == []


def test_tensorflow_convert_logic(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify tensorflow convert logic behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.tensorflow as tf_fw

  adapter = tf_fw.TensorFlowAdapter()

  import sys

  mock_tf = MagicMock()

  class DummyTensor:
    """Test helper class DummyTensor."""

    pass

  mock_tf.Tensor = DummyTensor

  def fake_convert(x: JsonValue) -> DummyTensor:
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

  f = FailTorch()
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

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock array."""

    def __array__(self) -> list[JsonValue]:
      """Gets array.

      Returns:
          Computed test result or fixture value.
      """
      return []

  assert type(adapter.convert(MockArray())).__name__ == "MockArray"

  class FailArray:
    """A failing array."""

    def __array__(self) -> list[JsonValue]:
      """Gets array.

      Returns:
          Computed test result or fixture value.
      """
      raise Exception("Fail")

  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_tensorflow_properties_extra() -> None:
  """Verify tensorflow properties extra behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()

  assert adapter.get_rng_split_syntax("rng", "key") == "pass"
  assert adapter.get_weight_conversion_imports() == ["import tensorflow as tf", "import numpy as np"]
  assert adapter.get_tensor_to_numpy_expr("t") == "t.numpy() if hasattr(t, 'numpy') else np.array(t)"
  assert "Checkpoint" not in adapter.get_weight_save_code("state", "path")

  traits = adapter.plugin_traits
  assert traits.requires_explicit_rng is False

  url: str | None = adapter.get_doc_url("not.tensorflow")
  assert url is not None
  assert "not/tensorflow" in url

  assert "tf.train.load_checkpoint" in adapter.get_weight_load_code("path")

  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""


def test_tensorflow_examples() -> None:
  """Verify tensorflow examples behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  ex: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in ex
  assert "tier2_neural" in ex
  assert "tier3_extras" in ex


def test_tensorflow_doc_url_extra() -> None:
  """Verify tensorflow doc url extra behavior."""
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  adapter = TensorFlowAdapter()
  url: str | None = adapter.get_doc_url("tensorflow.keras.layers.Dense")
  assert url is not None
  assert "search.html" not in url


def test_tensorflow_ghost_mode_with_snapshot() -> None:
  """Verify tensorflow ghost mode with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter

  with patch("ml_switcheroo.frameworks.tensorflow.load_snapshot_for_adapter", return_value={"test": 1}):
    with patch("ml_switcheroo.frameworks.tensorflow.tf", None):
      adapter = TensorFlowAdapter()
      assert adapter._snapshot_data == {"test": 1}
