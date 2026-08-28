"""Tests for Torch framework adapters."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.torch_io import TorchIOMixin
from ml_switcheroo.frameworks.torch_examples import get_torch_tiered_examples
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import InitMode


def test_torch_adapter_properties() -> None:
  """Test element."""
  adapter: TorchAdapter = TorchAdapter()

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

  with patch("ml_switcheroo.frameworks.torch.load_definitions", return_value={"test": MagicMock()}):
    adapter.definitions

  with patch.dict("sys.modules", {"torch": None}):
    adapter.convert([1, 2])

  with patch.dict("sys.modules", {"torch": MagicMock()}):
    import torch

    torch.tensor.return_value = "tensor"
    adapter.convert([1, 2])

    torch.tensor.side_effect = Exception("err")
    adapter.convert([1, 2])

  adapter.get_device_syntax("cpu", "1")
  adapter.get_device_syntax("cpu", "var")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path", "obj")
  adapter.get_serialization_syntax("save", "path", "obj")
  adapter.get_serialization_syntax("other", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("s", "path")

  adapter.apply_wiring({"mappings": {"test": {"api": "torch.test"}, "bad": {}, "none": None, "no_api": {"a": 1}}})
  adapter.get_doc_url("torch.add")
  adapter.get_doc_url("unknown")


def test_torch_ghost() -> None:
  """Test element."""
  with patch("ml_switcheroo.frameworks.torch.torch", None):
    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={}):
      adapter: TorchAdapter = TorchAdapter()
      assert adapter._mode == InitMode.GHOST


def test_torch_collect() -> None:
  """Test element."""
  adapter: TorchAdapter = TorchAdapter()
  adapter._snapshot_data = {}
  adapter._collect_ghost(SemanticTier.NEURAL)

  with patch("ml_switcheroo.frameworks.torch.torch", MagicMock()):
    adapter._collect_live(SemanticTier.LOSS)
    adapter._collect_live(SemanticTier.OPTIMIZER)
    adapter._collect_live(SemanticTier.ACTIVATION)
    adapter._collect_live(SemanticTier.LAYER)


def test_torch_examples() -> None:
  """Test element."""
  get_torch_tiered_examples()


def test_torch_io() -> None:
  """Test element."""
  mixin: TorchIOMixin = TorchIOMixin()
  mixin.get_serialization_imports()
  mixin.get_serialization_syntax("load", "path")
  mixin.get_serialization_syntax("save", "path", "obj")
  mixin.get_serialization_syntax("other", "path")
  mixin.get_weight_conversion_imports()
  mixin.get_weight_load_code("path")
  mixin.get_tensor_to_numpy_expr("t")
  mixin.get_weight_save_code("s", "path")


def test_torch_missed() -> None:
  """Test element."""
  adapter: TorchAdapter = TorchAdapter()
  adapter.get_tiered_examples()

  import numpy as np

  with patch.dict("sys.modules", {"torch": MagicMock()}):
    import torch

    torch.tensor.return_value = "mock_tensor"
    adapter.convert(np.array([1, 2]))
