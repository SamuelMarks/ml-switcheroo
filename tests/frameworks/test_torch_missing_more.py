def test_torch_adapter_coverage():
  import sys
  from ml_switcheroo.frameworks.base import InitMode, StandardCategory

  # mock torch
  class MockTorch:
    class nn:
      class functional:
        pass

      class modules:
        pass

    class optim:
      pass

  sys.modules["torch"] = MockTF = MockTorch()
  sys.modules["torch.nn"] = MockTorch.nn
  sys.modules["torch.nn.functional"] = MockTorch.nn.functional
  sys.modules["torch.optim"] = MockTorch.optim

  from ml_switcheroo.frameworks.torch import TorchAdapter

  adapter = TorchAdapter()
  adapter._mode = InitMode.LIVE

  # Run properties
  _ = adapter.unsafe_submodules
  _ = adapter.search_modules
  _ = adapter.import_alias
  _ = adapter.harness_imports
  _ = adapter.declared_magic_args
  _ = adapter.structural_traits
  _ = adapter.plugin_traits
  _ = adapter.definitions
  _ = adapter.rng_seed_methods

  # Run methods
  _ = adapter.get_harness_init_code()
  _ = adapter.get_to_numpy_code()
  _ = adapter.get_device_syntax("cuda")
  _ = adapter.get_device_syntax("cuda", "i")
  _ = adapter.get_device_syntax("cuda", "1")
  _ = adapter.get_device_check_syntax()
  _ = adapter.get_rng_split_syntax("rng", "key")
  _ = adapter.get_serialization_imports()
  _ = adapter.get_serialization_syntax("op", "f")
  _ = adapter.get_weight_conversion_imports()
  _ = adapter.get_weight_load_code("path")
  _ = adapter.get_tensor_to_numpy_expr("tensor")
  _ = adapter.get_weight_save_code("state", "path")
  _ = adapter.get_doc_url("torch.nn")
  _ = adapter.get_doc_url("other")
  _ = adapter.get_tiered_examples()

  # Collect API
  assert isinstance(adapter.collect_api(StandardCategory.LAYER), list)
  assert isinstance(adapter.collect_api(StandardCategory.LOSS), list)
  assert isinstance(adapter.collect_api(StandardCategory.OPTIMIZER), list)
  assert isinstance(adapter.collect_api(StandardCategory.ACTIVATION), list)

  # Force _collect_ghost
  adapter._mode = InitMode.GHOST
  adapter._snapshot_data = {}
  adapter.collect_api(StandardCategory.LAYER)
  adapter._snapshot_data = {
    "categories": {"layer": [{"api": "layer", "api_path": "layer", "name": "layer", "kind": "function", "args": []}]}
  }
  adapter.collect_api(StandardCategory.LAYER)
  _ = adapter.search_modules

  # Test apply_wiring
  snap = {"mappings": {"a": {"api": "torch.math.add"}, "b": {"api": "other"}}}
  adapter.apply_wiring(snap)

  del sys.modules["torch"]
  del sys.modules["torch.nn"]
  del sys.modules["torch.nn.functional"]
  del sys.modules["torch.optim"]


def test_torch_convert_and_init():
  import sys

  sys.modules["torch"] = None

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={}
  ):
    from ml_switcheroo.frameworks.torch import TorchAdapter

    adapter = TorchAdapter()
    assert adapter.convert(123) == 123


def test_torch_convert_more():
  import sys
  import numpy as np

  class MockTorch:
    @staticmethod
    def from_numpy(d):
      if d.shape == (2,):
        raise Exception()
      return d

    @staticmethod
    def tensor(d):
      if isinstance(d, list) and d == [3]:
        raise Exception()
      return d

  sys.modules["torch"] = MockTorch()
  sys.modules["numpy"] = np

  from ml_switcheroo.frameworks.torch import TorchAdapter

  adapter = TorchAdapter()

  adapter.convert(np.array([1]))
  adapter.convert(np.array([1, 2]))

  adapter.convert([1, 2])
  adapter.convert([3])
  adapter.convert("foo")

  del sys.modules["torch"]


def test_torch_import_error():
  import sys, importlib

  sys.modules["torch"] = None
  try:
    import ml_switcheroo.frameworks.torch as t

    importlib.reload(t)
  finally:
    del sys.modules["torch"]


def test_torch_init_no_torch():
  import sys, importlib

  sys.modules["torch"] = None
  try:
    import ml_switcheroo.frameworks.torch as t

    importlib.reload(t)
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={}
    ):
      t.TorchAdapter()
  finally:
    del sys.modules["torch"]
