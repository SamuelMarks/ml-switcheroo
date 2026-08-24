"""Tests for Numpy framework adapter."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.frameworks.numpy import NumpyAdapter


def test_numpy_adapter_properties():
  """Test element."""
  adapter = NumpyAdapter()

  adapter.import_alias
  adapter.import_namespaces
  adapter.supported_tiers
  adapter.structural_traits
  adapter.plugin_traits
  adapter.test_config
  adapter.harness_imports
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.declared_magic_args
  adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.numpy.load_definitions", return_value={"test": MagicMock()}):
    pass

  with patch.dict("sys.modules", {"numpy": None}):
    adapter.convert([1, 2])

  with patch.dict("sys.modules", {"numpy": MagicMock()}):
    import numpy as np

    np.array.return_value = "tensor"
    adapter.convert([1, 2])

    np.array.side_effect = Exception("err")
    adapter.convert([1, 2])

  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_serialization_syntax("load", "path")
  adapter.get_serialization_syntax("other", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({})
  adapter.get_doc_url("numpy.add")
  adapter.get_tiered_examples()


def test_numpy_convert_branches():
  """Test element."""
  adapter = NumpyAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class DetachDummy:
    def detach(self):
      class CpuDummy:
        def cpu(self):
          class NumpyDummy:
            def numpy(self):
              return "detached"

          return NumpyDummy()

      return CpuDummy()

  assert adapter.convert(DetachDummy()) == "detached"

  class DetachFail:
    def detach(self):
      raise Exception("fail")

  adapter.convert(DetachFail())

  class NumpyDummy:
    def numpy(self):
      return "numpy"

  assert adapter.convert(NumpyDummy()) == "numpy"

  class NumpyFail:
    def numpy(self):
      raise Exception("fail")

  adapter.convert(NumpyFail())

  class ArrayDummy:
    def __array__(self, dtype=None):
      return np.array([1], dtype=dtype)

  import numpy as np

  assert np.array_equal(adapter.convert(ArrayDummy()), np.array([1]))

  class ArrayFail:
    def __array__(self):
      raise Exception("fail")

  adapter.convert(ArrayFail())
