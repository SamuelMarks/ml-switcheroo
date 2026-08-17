"""Test suite for the Torch module."""

from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier
from unittest.mock import patch


def test_torch_import_exception():
  """Verifies the behavior when torch fails to import during module load."""
  from importlib import reload
  import ml_switcheroo.frameworks.torch as torch_module

  real_import = __import__

  def mock_import(name, *args, **kwargs):
    """Mocks python import builtin."""
    if name == "torch":
      raise Exception("Simulated Torch load failure")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    reload(torch_module)
    assert torch_module.torch is None
    assert torch_module.nn is None
    assert torch_module.optim is None

  reload(torch_module)  # restore


def test_torch_adapter_init():
  """Verifies the behavior of PyTorch adapter initialization."""
  adapter = TorchAdapter()
  assert adapter.display_name == "PyTorch"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 0
  assert adapter._mode in (InitMode.GHOST, InitMode.LIVE)


def test_torch_init_ghost_no_snapshot(monkeypatch):
  """Verifies the behavior of PyTorch initialization ghost no snapshot."""
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", None)
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", lambda _: {})
  with patch("logging.warning") as mock_warn:
    adapter = TorchAdapter()
    assert adapter._mode == InitMode.GHOST
    mock_warn.assert_called_once_with("PyTorch not installed and no snapshot found. Scanning unavailable.")


def test_torch_init_ghost(monkeypatch):
  """Verifies the behavior of PyTorch initialization ghost."""
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", None)
  adapter = TorchAdapter()
  assert adapter._mode == InitMode.GHOST


def test_torch_definitions_fallback(monkeypatch):
  """Verifies missing defs are populated."""
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.load_definitions", lambda _: {})
  adapter = TorchAdapter()
  defs = adapter.definitions
  assert "ReLU" in defs
  assert "Linear" in defs
  assert "Conv2d" in defs


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

  # Test loops in apply_wiring (which does nothing currently except setdefault)
  snapshot2 = {"mappings": {"op1": {"api": "torch.Tensor"}}}
  adapter.apply_wiring(snapshot2)
  assert "op1" in snapshot2["mappings"]


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

  # When torch is mocked as None
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

  # Test successful convert with real torch
  import torch
  import numpy as np

  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", torch)
  res = adapter.convert([1, 2])
  assert isinstance(res, torch.Tensor)

  # Test with numpy
  arr_np = np.array([1, 2])
  res_np = adapter.convert(arr_np)
  # the test framework already has torch imported globally probably, but let's check it's tensor
  assert isinstance(res_np, torch.Tensor)

  # Test with mock torch that throws error on tensor conversion
  class FailTorch:
    """Mock module for failure testing."""

    def tensor(self, data):
      """Mock tensor func."""
      if isinstance(data, (list, tuple)):
        raise Exception("Simulate fail list")
      return "failed_to_tensor"

    def from_numpy(self, data):
      """Mock from_numpy func."""
      raise Exception("Simulate fail")

  import sys

  monkeypatch.setitem(sys.modules, "torch", FailTorch())
  monkeypatch.setitem(sys.modules, "numpy", np)  # keep np
  monkeypatch.setattr("ml_switcheroo.frameworks.torch.torch", FailTorch())
  res_list = adapter.convert([1, 2])
  assert res_list == [1, 2]
  # fallback to tensor which succeeds
  assert adapter.convert(arr_np) == "failed_to_tensor"

  # test missing modules completely
  monkeypatch.setitem(sys.modules, "torch", None)
  monkeypatch.setitem(sys.modules, "numpy", None)
  monkeypatch.setattr("builtins.__import__", lambda *a, **k: exec("raise ImportError('Missing')"))
  assert adapter.convert([1, 2]) == [1, 2]


def test_torch_collect_ghost():
  """Verifies collect_ghost."""
  adapter = TorchAdapter()
  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.LOSS) == []

  adapter._snapshot_data = {
    "categories": {
      "loss": [{"api_path": "torch.nn.MSELoss", "name": "MSELoss", "kind": "Loss", "group": "Loss", "params": []}]
    }
  }
  refs = adapter._collect_ghost(SemanticTier.LOSS)
  assert len(refs) == 1
  assert refs[0].name == "MSELoss"


def test_torch_collect_live(monkeypatch):
  """Verifies collect_live routing."""
  adapter = TorchAdapter()

  adapter._scan_losses = lambda: ["loss"]
  adapter._scan_optimizers = lambda: ["opt"]
  adapter._scan_activations = lambda: ["act"]
  adapter._scan_layers = lambda: ["layer"]

  assert adapter._collect_live(SemanticTier.LOSS) == ["loss"]
  assert adapter._collect_live(SemanticTier.OPTIMIZER) == ["opt"]
  assert adapter._collect_live(SemanticTier.ACTIVATION) == ["act"]
  assert adapter._collect_live(SemanticTier.LAYER) == ["layer"]
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


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


def test_torch_missing_coverage():
  """Verifies missing coverage methods for TorchAdapter."""
  adapter = TorchAdapter()

  # Traits
  assert adapter.plugin_traits.has_numpy_compatible_arrays is False

  # RNG split
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  # Serialization
  assert "import torch" in adapter.get_serialization_imports()
  assert "torch.save(obj, f)" == adapter.get_serialization_syntax("save", "f", "obj")
  assert "torch.load(f)" == adapter.get_serialization_syntax("load", "f")
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.get_serialization_syntax("unknown", "f") == ""

  # Weight Conversion
  assert "import torch" in adapter.get_weight_conversion_imports()
  assert "torch.load" in adapter.get_weight_load_code("path")
  assert "var.detach().cpu().numpy()" in adapter.get_tensor_to_numpy_expr("var")
  assert "torch.save" in adapter.get_weight_save_code("state", "path")
