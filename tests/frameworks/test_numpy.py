"""Test suite for the Numpy module."""

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_numpy_adapter_init():
  """Verifies the behavior of NumPy adapter initialization."""
  adapter = NumpyAdapter()
  assert adapter.display_name == "NumPy"


def test_numpy_import_alias():
  """Verifies the behavior of NumPy import alias."""
  adapter = NumpyAdapter()
  assert adapter.import_alias == ("numpy", "np")


def test_numpy_import_namespaces():
  """Verifies the behavior of NumPy import namespaces."""
  adapter = NumpyAdapter()
  ns = adapter.import_namespaces
  assert "numpy" in ns


def test_numpy_test_config():
  """Verifies the behavior of NumPy test configuration."""
  adapter = NumpyAdapter()
  config = adapter.test_config
  assert "import numpy as np" in config["import"]


def test_numpy_harness_imports():
  """Verifies the behavior of NumPy harness imports."""
  adapter = NumpyAdapter()
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""


def test_numpy_get_to_numpy_code():
  """Verifies the behavior of NumPy get to NumPy code."""
  adapter = NumpyAdapter()
  assert "isinstance(obj, np.ndarray)" in adapter.get_to_numpy_code()


def test_numpy_supported_tiers():
  """Verifies the behavior of NumPy supported tiers."""
  adapter = NumpyAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_numpy_declared_magic_args():
  """Verifies the behavior of NumPy declared magic arguments."""
  adapter = NumpyAdapter()
  assert adapter.declared_magic_args == []


def test_numpy_structural_traits():
  """Verifies the behavior of NumPy structural traits."""
  adapter = NumpyAdapter()
  traits = adapter.structural_traits
  assert traits.auto_strip_magic_args


def test_numpy_definitions():
  """Verifies the behavior of NumPy definitions."""
  adapter = NumpyAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_numpy_rng_seed_methods():
  """Verifies the behavior of NumPy rng seed methods."""
  adapter = NumpyAdapter()
  assert "seed" in adapter.rng_seed_methods


def test_numpy_device_syntax():
  """Verifies the behavior of NumPy device syntax."""
  adapter = NumpyAdapter()
  assert adapter.get_device_syntax("cuda") == "'cpu'"


def test_numpy_serialization_syntax():
  """Verifies the behavior of NumPy serialization syntax."""
  adapter = NumpyAdapter()
  assert "import numpy as np" in adapter.get_serialization_imports()
  assert "np.save" in adapter.get_serialization_syntax("save", "f", "obj")
  assert "np.load" in adapter.get_serialization_syntax("load", "f")


def test_numpy_apply_wiring():
  """Verifies the behavior of NumPy apply wiring."""
  adapter = NumpyAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}


def test_numpy_doc_url():
  """Verifies the behavior of NumPy documentation URL."""
  adapter = NumpyAdapter()
  assert "numpy.org" in adapter.get_doc_url("numpy.abs")


class DummyTensor:
  """Dummy Tensor class for testing purposes."""

  def __init__(self, data):
    """Initializes the DummyTensor instance."""
    self.data = data

  def detach(self):
    """Mock implementation of detach."""
    return self

  def cpu(self):
    """Mock implementation of cpu."""
    return self

  def numpy(self):
    """Mock implementation of NumPy."""
    return self.data


def test_numpy_convert():
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

    def __array__(self, dtype=None, copy=None):
      """Arr."""
      return [3]

  converted_arr = adapter.convert(HasArray())
  assert isinstance(converted_arr, HasArray)


def test_numpy_tiered_examples():
  """Verifies the behavior of NumPy tiered examples."""
  adapter = NumpyAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier3_extras" in examples
