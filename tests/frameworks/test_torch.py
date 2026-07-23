"""Test suite for the Torch module."""

from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier
from unittest.mock import patch


def test_torch_adapter_init():
  """Verifies the behavior of PyTorch adapter initialization."""
  adapter = TorchAdapter()
  assert adapter.display_name == "PyTorch"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 0
  assert adapter._mode in (InitMode.GHOST, InitMode.LIVE)


def test_torch_init_ghost(monkeypatch):
  """Verifies the behavior of PyTorch initialization ghost."""
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", None)
  adapter = TorchAdapter()
  assert adapter._mode == InitMode.GHOST


def test_torch_properties():
  """Verifies the behavior of PyTorch properties."""
  adapter = TorchAdapter()
  assert adapter.import_alias == ("torch", "torch")
  ns = adapter.import_namespaces
  assert "torch" in ns
  assert "torch.nn" in ns
  assert "torch.optim" in ns
  config = adapter.test_config
  assert "import torch" in config["import"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "hasattr(obj, 'detach')" in adapter.get_to_numpy_code()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  traits = adapter.structural_traits
  assert traits.module_base == "torch.nn.Module"
  assert traits.forward_method == "forward"
  assert traits.requires_super_init
  defs = adapter.definitions
  assert isinstance(defs, dict)
  assert "manual_seed" in adapter.rng_seed_methods


def test_torch_apply_wiring():
  """Verifies the behavior of PyTorch apply wiring."""
  adapter = TorchAdapter()
  snapshot = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot


def test_torch_device_syntax():
  """Verifies the behavior of PyTorch device syntax."""
  adapter = TorchAdapter()
  assert "torch.device(cuda)" == adapter.get_device_syntax("cuda")
  assert "torch.device(cpu)" == adapter.get_device_syntax("cpu")
  assert "torch.device(cuda, 1)" == adapter.get_device_syntax("cuda", "1")


def test_torch_device_check_syntax():
  """Verifies the behavior of PyTorch device check syntax."""
  adapter = TorchAdapter()
  assert "torch.cuda.is_available()" in adapter.get_device_check_syntax()


def test_torch_doc_url():
  """Verifies the behavior of PyTorch documentation URL."""
  adapter = TorchAdapter()
  url = adapter.get_doc_url("torch.abs")
  assert "torch.abs.html" in url
  url_init = adapter.get_doc_url("torch.nn.init.uniform_")
  assert "nn.init.html" in url_init


def test_torch_convert(monkeypatch):
  """Verifies the behavior of PyTorch convert."""
  adapter = TorchAdapter()
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", None)
  assert adapter.convert("test") == "test"

  class DummyNumpy:
    """Dummy."""

    def __init__(self):
      """Init."""
      pass

  arr = DummyNumpy()
  converted = adapter.convert(arr)
  assert converted is arr


@patch("ml_switcheroo.frameworks.torch_examples.get_torch_tiered_examples")
def test_torch_tiered_examples(mock_examples):
  """Verifies the behavior of PyTorch tiered examples."""
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = TorchAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_torch_init_live(monkeypatch):
  """Verifies the behavior of PyTorch initialization live."""
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", True)
  adapter = TorchAdapter()
  assert adapter._mode == InitMode.LIVE
