"""Test suite for the Numpy module."""

import typing

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
  ns: typing.Any = adapter.import_namespaces
  assert "numpy" in ns


def test_numpy_test_config() -> None:
  """Docstring."""
  adapter = NumpyAdapter()
  config: dict[str, typing.Any] = adapter.test_config
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
  tiers: set[SemanticTier] = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_numpy_declared_magic_args() -> None:
  """Verifies the behavior of NumPy declared magic arguments."""
  adapter = NumpyAdapter()
  assert adapter.declared_magic_args == []


def test_numpy_structural_traits() -> None:
  """Verifies the behavior of NumPy structural traits."""
  adapter = NumpyAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.auto_strip_magic_args


def test_numpy_definitions() -> None:
  """Verifies the behavior of NumPy definitions."""
  adapter = NumpyAdapter()
  defs: typing.Any = adapter.definitions
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
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}


def test_numpy_doc_url() -> None:
  """Verifies the behavior of NumPy documentation URL."""
  adapter = NumpyAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("numpy.abs")
  assert url is not None
  assert "numpy.org" in url


class DummyTensor:
  """Docstring."""

  def __init__(self, data: typing.Any) -> None:
    """Initializes the DummyTensor instance."""
    self.data = data

  def detach(self) -> "DummyTensor":
    """Mock implementation of detach."""
    return self

  def cpu(self) -> "DummyTensor":
    """Mock implementation of cpu."""
    return self

  def numpy(self) -> typing.Any:
    """Mock implementation of NumPy."""
    return self.data


def test_numpy_convert() -> None:
  """Verifies the behavior of NumPy convert."""
  adapter = NumpyAdapter()
  converted: typing.Any = adapter.convert([1, 2])
  assert isinstance(converted, list)
  converted_dict: typing.Any = adapter.convert({"a": 1})
  assert isinstance(converted_dict, dict)
  tensor = DummyTensor([1, 2])
  converted_tensor: typing.Any = adapter.convert(tensor)
  assert isinstance(converted_tensor, list)

  class HasArray:
    """Has."""

    def __array__(self, dtype: typing.Any = None, copy: typing.Any = None) -> list[int]:
      """Arr."""
      return [3]

  converted_arr: typing.Any = adapter.convert(HasArray())
  assert isinstance(converted_arr, HasArray)


def test_numpy_tiered_examples() -> None:
  """Verifies the behavior of NumPy tiered examples."""
  adapter = NumpyAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier3_extras" in examples


# --- Merged from test_numpy_extra.py ---


def test_numpy_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import builtins
  import importlib
  from unittest.mock import patch

  import ml_switcheroo.frameworks.numpy as np_fw

  orig_import = builtins.__import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Docstring."""
    if name == "numpy":
      raise ImportError("no numpy")
    return orig_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import):
    importlib.reload(np_fw)
    assert getattr(np_fw, "np", None) is None

  importlib.reload(np_fw)


def test_numpy_convert_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
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
      """Mocks detach."""
      return self

    def cpu(self) -> "MockTorch":
      """Mocks cpu."""
      return self

    def numpy(self) -> str:
      """Mocks numpy."""
      return "numpy_tensor"

  assert adapter.convert(MockTorch()) == "numpy_tensor"

  class FailTorch:
    """A fake failing torch tensor."""

    def detach(self) -> "FailTorch":
      """Mocks detach."""
      raise Exception("Fail")

  f = FailTorch()
  assert adapter.convert(f) is f

  class MockTF:
    """A mock TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy."""
      return "tf_tensor"

  assert adapter.convert(MockTF()) == "tf_tensor"

  class FailTF:
    """A failing TF tensor."""

    def numpy(self) -> str:
      """Mocks numpy."""
      raise Exception("Fail")

  f2 = FailTF()
  assert adapter.convert(f2) is f2

  class MockArray:
    """A mock numpy array interface."""

    def __array__(self) -> list[typing.Any]:
      """Gets array."""
      return []

  assert adapter.convert(MockArray()) == "mock_array"

  class FailArray:
    """A failing array."""

    def __array__(self) -> list[typing.Any]:
      """Gets array."""
      return []  # trigger the lambda then raise

  def failing_array(x: typing.Any) -> str:
    """Mocks numpy array creation."""
    if isinstance(x, FailArray):
      raise Exception("Fail")
    return "mock_array2"

  mock_np2 = type("MockNP2", (), {"array": failing_array})
  monkeypatch.setattr(np_fw, "np", mock_np2)

  assert adapter.convert(MockArray()) == "mock_array2"
  f3 = FailArray()
  assert adapter.convert(f3) is f3


def test_numpy_properties() -> None:
  """Docstring."""
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

  traits: typing.Any = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True


def test_numpy_doc_url_extra() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert adapter.get_doc_url("numpy.array") == "https://numpy.org/doc/stable/reference/generated/numpy.array.html"


def test_numpy_test_config_extra() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "import numpy as np" in adapter.test_config["import"]


def test_numpy_get_to_numpy_code_extra() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "if isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()


def test_numpy_get_tiered_examples() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.numpy import NumpyAdapter

  adapter = NumpyAdapter()
  assert "tier1_math" in adapter.get_tiered_examples()
  assert "tier2_neural" in adapter.get_tiered_examples()
  assert "tier3_extras" in adapter.get_tiered_examples()
