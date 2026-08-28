"""Test suite for the Numpy module."""

import typing
from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


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
  """Verifies the behavior of NumPy test configuration."""
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
  """Dummy Tensor class for testing purposes."""

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
