import pytest
from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.core.ghost import GhostRef
import numpy as np


def test_numpy_adapter():
  adapter = NumpyAdapter()
  assert adapter.display_name == "NumPy"

  assert "numpy" in adapter.search_modules
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("numpy", "np")

  ns = adapter.import_namespaces
  assert "numpy" in ns
  assert ns["numpy"].tier == SemanticTier.ARRAY_API

  heuristics = adapter.discovery_heuristics
  assert "extras" in heuristics

  tc = adapter.test_config
  assert "import" in tc

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() != ""
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []

  traits = adapter.structural_traits
  assert traits.auto_strip_magic_args

  pt = adapter.plugin_traits
  assert pt.has_numpy_compatible_arrays

  defs = adapter.definitions
  assert isinstance(defs, dict)

  assert "seed" in adapter.rng_seed_methods

  # Discovery
  assert adapter.collect_api(StandardCategory.LAYER) == []

  # syntax
  assert adapter.get_device_syntax("cuda") == "'cpu'"
  assert adapter.get_device_check_syntax() == "False"
  assert adapter.get_rng_split_syntax("r", "k") == "pass"

  assert adapter.get_serialization_imports() == ["import numpy as np"]
  assert adapter.get_serialization_syntax("save", "'file.npy'", "model") == "np.save(file='file.npy', arr=model)"
  assert adapter.get_serialization_syntax("load", "'file.npy'") == "np.load(file='file.npy')"
  assert adapter.get_serialization_syntax("unknown", "'file.npy'") == ""

  assert adapter.get_weight_conversion_imports() == ["import numpy as np"]
  assert "np.load(path, allow_pickle=True)" in adapter.get_weight_load_code("path")
  assert "tensor" in adapter.get_tensor_to_numpy_expr("tensor")
  assert "np.savez_compressed" in adapter.get_weight_save_code("state", "path")

  adapter.apply_wiring({})
  assert adapter.get_doc_url("numpy.abs") == "https://numpy.org/doc/stable/reference/generated/numpy.abs.html"

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples

  # convert
  res = adapter.convert([1, 2, 3])
  assert isinstance(res, list)  # Oh wait, list to list or tuple, recursively
  assert isinstance(adapter.convert(np.array([1, 2, 3])), np.ndarray)

  assert adapter.convert("test") == "test"

  class FakeTensor:
    def numpy(self):
      return np.array([1])

  assert isinstance(adapter.convert(FakeTensor()), np.ndarray)

  class FakeTensorDetach:
    def detach(self):
      class Cpu:
        def cpu(self):
          return FakeTensor()

      return Cpu()

  assert isinstance(adapter.convert(FakeTensorDetach()), np.ndarray)

  assert isinstance(adapter.convert({"a": [1]}), dict)
  assert isinstance(adapter.convert(([1],)), tuple)

  class FakeArray:
    def __array__(self):
      return np.array([1])

  assert isinstance(adapter.convert(FakeArray()), np.ndarray)

  # test Exceptions inside convert
  class BrokenTensor:
    def numpy(self):
      raise ValueError("broken")

  class BrokenDetach:
    def detach(self):
      raise ValueError("broken")

  class BrokenArray:
    def __array__(self):
      raise ValueError("broken")

  assert adapter.convert(BrokenTensor()) is not None
  assert adapter.convert(BrokenDetach()) is not None
  assert adapter.convert(BrokenArray()) is not None
