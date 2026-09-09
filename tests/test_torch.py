"""Module docstring."""

from typing import Any
import numpy as np

from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_no_torch_module(monkeypatch):
  """Docstring."""
  import sys

  monkeypatch.setitem(sys.modules, "torch", None)
  import importlib

  import ml_switcheroo.frameworks.torch as torch_mod

  try:
    importlib.reload(torch_mod)
  except Exception:
    pass


def test_torch_get_doc_url_init():
  """Docstring."""
  fw = TorchAdapter()
  assert "nn.init.html" in fw.get_doc_url("torch.nn.init.constant_")


def test_torch_convert_numpy_exception(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch = MagicMock()
  mock_tensor_cls = type("Tensor", (), {})
  mock_torch.Tensor = mock_tensor_cls

  def mock_from_numpy(data):
    """Docstring."""
    raise ValueError("mock error")

  mock_torch.from_numpy = mock_from_numpy
  mock_torch.tensor.return_value = mock_tensor_cls()
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res = fw.convert(np.array([1, 2]))
  assert isinstance(res, mock_tensor_cls)


def test_torch_convert_list_exception(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch = MagicMock()

  def mock_tensor(data):
    """Docstring."""
    raise ValueError("mock error")

  mock_torch.tensor = mock_tensor
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res = fw.convert([1, 2, 3])
  # Should fall through and return the list
  assert res == [1, 2, 3]


def test_torch_collect_ghost_no_snapshot():
  """Docstring."""
  fw = TorchAdapter()
  fw._snapshot_data = None
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  assert fw._collect_ghost(SemanticTier.LAYER) == []

  fw._snapshot_data = {"categories": {"layer": [{"api_path": "test", "name": "Test", "kind": "function"}]}}
  res = fw._collect_ghost(SemanticTier.LAYER)
  assert len(res) == 1
  assert res[0].api_path == "test"


def test_torch_collect_live_layer():
  """Docstring."""
  fw = TorchAdapter()
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  def dummy_scan_layers():
    """Docstring."""
    return ["layer1"]

  fw._scan_layers = dummy_scan_layers
  res = fw._collect_live(SemanticTier.LAYER)
  assert res == ["layer1"]


def test_torch_import_fail_in_convert(monkeypatch):
  """Docstring."""
  import builtins

  fw = TorchAdapter()
  original_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    """Docstring."""
    if name == "torch":
      raise ImportError("mock error")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res = fw.convert([1, 2, 3])
  assert res == [1, 2, 3]


def test_torch_get_device_syntax():
  """Docstring."""
  fw = TorchAdapter()
  # Hit 221
  res = fw.get_device_syntax("xpu")
  assert "xpu" in res

  res = fw.get_device_syntax("cuda", "1")
  assert "1" in res


def test_torch_convert_numpy_fallback(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Docstring."""

    def from_numpy(self, data):
      """Docstring."""
      raise ValueError()

    def tensor(self, data):
      """Docstring."""
      return "fallback"

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert(np.array([1, 2]))
  assert res == "fallback"


def test_torch_convert_list_fallback(monkeypatch):
  """Docstring."""
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Docstring."""

    def tensor(self, data):
      """Docstring."""
      raise ValueError()

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert([1, 2])
  assert res == [1, 2]


def test_torch_definitions_fallback(monkeypatch):
  """Docstring."""
  import ml_switcheroo.frameworks.torch as torch_mod

  monkeypatch.setattr(torch_mod, "load_definitions", lambda fw: {})
  fw = TorchAdapter()
  if callable(fw.definitions):
    defs = fw.definitions()
  else:
    defs = fw.definitions
  assert "ReLU" in defs
  assert "Linear" in defs
  assert "Conv2d" in defs


def test_torch_import_success() -> None:
  """Test module-level import when torch is available."""
  import importlib
  import sys
  from unittest.mock import MagicMock, patch

  mock_torch = MagicMock()
  with patch.dict(
    sys.modules,
    {
      "torch": mock_torch,
      "torch.nn": MagicMock(),
      "torch.nn.functional": MagicMock(),
      "torch.optim": MagicMock(),
    },
  ):
    import ml_switcheroo.frameworks.torch as mod

    importlib.reload(mod)
    adapter = mod.TorchAdapter()
    assert adapter._mode == mod.InitMode.LIVE
  importlib.reload(mod)


def test_torch_properties_and_methods() -> None:
  """Test properties and remaining methods on TorchAdapter."""
  from unittest.mock import MagicMock, patch
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  adapter: TorchAdapter = TorchAdapter()
  assert adapter.import_alias == ("torch", "torch")
  assert "torch" in adapter.import_namespaces
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "detach" in adapter.get_to_numpy_code()
  assert adapter.structural_traits.module_base == "torch.nn.Module"
  assert adapter.plugin_traits.has_numpy_compatible_arrays is False
  assert "manual_seed" in adapter.rng_seed_methods
  assert adapter.declared_magic_args == []
  assert adapter.get_rng_split_syntax("r", "k") == "pass"
  assert adapter.get_device_check_syntax() == "torch.cuda.is_available()"
  assert adapter.convert(42) == 42
  assert "tier1_math" in adapter.get_tiered_examples()

  s: dict[str, Any] = {}
  adapter.apply_wiring(s)
  assert "mappings" in s

  assert "generated/torch.add.html" in (adapter.get_doc_url("torch.add") or "")

  all_defs = {
    "ReLU": MagicMock(),
    "Linear": MagicMock(),
    "Conv2d": MagicMock(),
    "Conv1d": MagicMock(),
    "Conv3d": MagicMock(),
    "ConvTranspose2d": MagicMock(),
  }
  with patch("ml_switcheroo.frameworks.torch.load_definitions", return_value=all_defs):
    defs = adapter.definitions
    assert "ReLU" in defs

  assert adapter._collect_live(SemanticTier.LOSS) == []
  assert adapter._collect_live(SemanticTier.OPTIMIZER) == []
  assert adapter._collect_live(SemanticTier.ACTIVATION) == []
  assert adapter._collect_live(SemanticTier.LAYER) == []
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_torch_ghost_init_modes() -> None:
  """Test ghost mode initialization with and without snapshot data."""
  from unittest.mock import patch
  from ml_switcheroo.frameworks.base import InitMode

  with patch("ml_switcheroo.frameworks.torch.torch", None):
    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={}):
      adapter_empty: TorchAdapter = TorchAdapter()
      assert adapter_empty._mode == InitMode.GHOST
      assert adapter_empty._snapshot_data == {}

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={"categories": {}}):
      adapter_snap: TorchAdapter = TorchAdapter()
      assert adapter_snap._mode == InitMode.GHOST
      assert "categories" in adapter_snap._snapshot_data
