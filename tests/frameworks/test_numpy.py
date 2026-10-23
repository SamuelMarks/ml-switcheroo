"""Test suite for the Numpy module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.numpy import NumpyAdapter


def test_numpy_adapter_init() -> None:
  """Verifies the behavior of NumPy adapter initialization."""
  adapter = NumpyAdapter()
  assert adapter.display_name == "numpy"


def test_numpy_import_alias() -> None:
  """Verifies the behavior of NumPy import alias."""
  adapter = NumpyAdapter()
  assert adapter.import_alias == ("numpy", "np")


def test_numpy_import_namespaces() -> None:
  """Verifies the behavior of NumPy import namespaces."""
  adapter = NumpyAdapter()
  ns = adapter.import_namespaces
  assert "numpy" in ns


def test_numpy_test_config() -> None:
  """Verify numpy test config behavior."""
  adapter = NumpyAdapter()
  config = adapter.test_config
  assert "import numpy as np" in config["import"]


def test_numpy_harness_imports() -> None:
  """Verifies the behavior of NumPy harness imports."""
  adapter = NumpyAdapter()
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""


def test_numpy_get_to_numpy_code() -> None:
  """Verifies the behavior of NumPy get to NumPy code."""
  adapter = NumpyAdapter()
  assert "isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()


def test_numpy_supported_tiers() -> None:
  """Verifies the behavior of NumPy supported tiers."""
  adapter = NumpyAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_numpy_declared_magic_args() -> None:
  """Verifies the behavior of NumPy declared magic arguments."""
  adapter = NumpyAdapter()
  assert adapter.declared_magic_args == []


def test_numpy_structural_traits() -> None:
  """Verifies the behavior of NumPy structural traits."""
  adapter = NumpyAdapter()
  traits = adapter.structural_traits
  assert traits.auto_strip_magic_args


def test_numpy_definitions() -> None:
  """Verifies the behavior of NumPy definitions."""
  adapter = NumpyAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_numpy_rng_seed_methods() -> None:
  """Verifies the behavior of NumPy rng seed methods."""
  adapter = NumpyAdapter()
  assert "seed" in adapter.rng_seed_methods


def test_numpy_device_syntax() -> None:
  """Verifies the behavior of NumPy device syntax."""
  adapter = NumpyAdapter()
  assert adapter.get_device_syntax("cuda") == "'cpu'"


def test_numpy_serialization_syntax() -> None:
  """Verifies the behavior of NumPy serialization syntax."""
  adapter = NumpyAdapter()
  assert "import numpy as np" in adapter.get_serialization_imports()
  assert "np.save" in adapter.get_serialization_syntax("save", "f", "obj")
  assert "np.load" in adapter.get_serialization_syntax("load", "f")


def test_numpy_apply_wiring() -> None:
  """Verifies the behavior of NumPy apply wiring."""
  adapter = NumpyAdapter()
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert snapshot == {}


def test_numpy_doc_url() -> None:
  """Verifies the behavior of NumPy documentation URL."""
  adapter = NumpyAdapter()
  url: str | None = adapter.get_doc_url("numpy.abs")
  assert url is not None
  assert "numpy.org" in url


class DummyTensor:
  """Test helper class DummyTensor."""

  def __init__(self, data: list[int]) -> None:
    """Initializes the DummyTensor instance.

    Args:
        data: The data (list[int]) for init.
    """
    self.data = data

  def detach(self) -> "DummyTensor":
    """Mock implementation of detach.

    Returns:
        Resulting `'DummyTensor'` from detach.
    """
    return self

  def cpu(self) -> "DummyTensor":
    """Mock implementation of cpu.

    Returns:
        Resulting `'DummyTensor'` from cpu.
    """
    return self

  def numpy(self) -> list[int]:
    """Mock implementation of NumPy.

    Returns:
        Resulting `list[int]` from numpy.
    """
    return self.data


def test_numpy_convert() -> None:
  """Verifies the behavior of NumPy convert."""
  adapter = NumpyAdapter()
  converted = adapter.convert([1, 2])
  assert isinstance(converted, list)
  converted_dict = adapter.convert({"a": 1})
  assert isinstance(converted_dict, dict)
  tensor = DummyTensor([1, 2])
  converted_tensor = adapter.convert(tensor)
  assert isinstance(converted_tensor, list)

  class HasArray:
    """Has."""

    def __array__(self, dtype: str | None = None, copy: bool | None = None) -> list[int]:
      """Arr.

      Args:
          dtype: The dtype (str | None) for array.
          copy: The copy (bool | None) for array.

      Returns:
          Resulting `list[int]` from array.
      """
      return [3]

  converted_arr = adapter.convert(HasArray())
  assert isinstance(converted_arr, HasArray)


def test_numpy_tiered_examples() -> None:
  """Verifies the behavior of NumPy tiered examples."""
  adapter = NumpyAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier3_extras" in examples


def test_numpy_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify numpy init missing behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import builtins
  import importlib
  import sys
  import types
  from unittest.mock import patch

  import ml_switcheroo.frameworks.numpy as np_fw

  numpy_modules = {k: v for k, v in sys.modules.items() if k == "numpy" or k.startswith("numpy.")}
  orig_import = builtins.__import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
    """Execute mock import helper logic.

    Args:
        name: Identifier or operation name string.
        args: The args (JsonValue) for mock import.
        kwargs: The kwargs (JsonValue) for mock import.

    Returns:
        Resulting `types.ModuleType` from mock import.
    """
    if name == "numpy":
      raise ImportError("no numpy")
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": orig_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  try:
    with patch("builtins.__import__", side_effect=mock_import):
      importlib.reload(np_fw)
      assert getattr(np_fw, "np", None) is None
  finally:
    sys.modules.update(numpy_modules)
    importlib.reload(np_fw)
    assert getattr(np_fw, "np", None) is not None


def test_numpy_convert_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify numpy convert extra behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import importlib

  import ml_switcheroo.frameworks.numpy as np_fw

  importlib.reload(np_fw)

  mock_np = type("MockNP", (), {"array": lambda x: "mock_array"})
  monkeypatch.setattr(np_fw, "np", mock_np)

  adapter = np_fw.NumpyAdapter()

  assert adapter.convert([1, 2, 3]) == [1, 2, 3]
  assert adapter.convert({"a": 1}) == {"a": 1}

  class MockTorch:
    """A mock Torch tensor."""

    def detach(self) -> "MockTorch":
      """Mocks detach.

      Returns:
          Resulting `'MockTorch'` from detach.
      """
      return self

    def cpu(self) -> "MockTorch":
      """Mocks cpu.

      Returns:
          Resulting `'MockTorch'` from cpu.
      """
      return self

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Resulting `str` from numpy.
      """
      return "numpy_tensor"

  assert adapter.convert(MockTorch()) == "numpy_tensor"

  class FailTorch:
    """A fake failing torch tensor."""

    def detach(self) -> "FailTorch":
      """Mocks detach.

      Returns:
          Resulting `'FailTorch'` from detach.
      """
      raise RuntimeError("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Resulting `str` from numpy.
      """
      return "tf_tensor"

  assert adapter.convert(MockTF()) == "tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy.

      Returns:
          Resulting `str` from numpy.
      """
      raise RuntimeError("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock numpy array interface."""

    def __array__(self) -> list[JsonValue]:
      """Gets array.

      Returns:
          Resulting `list[JsonValue]` from array.
      """
      return []

  assert adapter.convert(MockArray()) == "mock_array"

  class FailArray:
    """A failing array."""

    def __array__(self) -> list[JsonValue]:
      """Gets array.

      Returns:
          Resulting `list[JsonValue]` from array.
      """
      return []  # trigger the lambda then raise

  def failing_array(x: FailArray | MockArray | JsonValue) -> str:
    """Mocks numpy array creation.

    Args:
        x: The x (FailArray | MockArray | JsonValue) for failing array.

    Returns:
        Resulting `str` from failing array.
    """
    if isinstance(x, FailArray):
      raise RuntimeError("Fail")
    return "mock_array2"

  mock_np2 = type("MockNP2", (), {"array": failing_array})
  monkeypatch.setattr(np_fw, "np", mock_np2)

  assert adapter.convert(MockArray()) == "mock_array2"
  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_numpy_properties() -> None:
  """Verify numpy properties behavior."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()

  assert adapter.get_device_syntax("cpu") == "'cpu'"
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"
  assert adapter.get_serialization_imports() == ["import numpy as np"]
  assert adapter.get_serialization_syntax("save", "file", "obj") == "np.save(file=file, arr=obj)"
  assert adapter.get_serialization_syntax("load", "file") == "np.load(file=file)"
  assert adapter.get_serialization_syntax("invalid", "file") == ""
  assert adapter.get_serialization_syntax("save", "file", None) == ""
  assert adapter.get_weight_conversion_imports() == ["import numpy as np"]
  assert "loaded = np.load(path, allow_pickle=True)" in adapter.get_weight_load_code("path")
  assert adapter.get_tensor_to_numpy_expr("t") == "t"
  assert adapter.get_weight_save_code("state", "path") == "np.savez_compressed(path, **state)"

  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True


def test_numpy_doc_url_extra() -> None:
  """Verify numpy doc url extra behavior."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert adapter.get_doc_url("numpy.array") == "https://numpy.org/doc/stable/reference/generated/numpy.array.html"


def test_numpy_test_config_extra() -> None:
  """Verify numpy test config extra behavior."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "import numpy as np" in adapter.test_config["import"]


def test_numpy_get_to_numpy_code_extra() -> None:
  """Verify numpy get to numpy code extra behavior."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "if isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()


def test_numpy_get_tiered_examples() -> None:
  """Verify numpy get tiered examples behavior."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "tier1_math" in adapter.get_tiered_examples()
  assert "tier2_neural" in adapter.get_tiered_examples()
  assert "tier3_extras" in adapter.get_tiered_examples()
