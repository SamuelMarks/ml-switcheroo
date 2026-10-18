"""Test suite for torch."""

import typing


import numpy as np

from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_no_torch_module(monkeypatch: typing.Any) -> None:
  """Verify torch no torch module behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys

  monkeypatch.setitem(sys.modules, "torch", None)
  import importlib

  import ml_switcheroo.frameworks.torch as torch_mod

  importlib.reload(torch_mod)
  assert torch_mod.torch is None


def test_torch_get_doc_url_init() -> None:
  """Verify torch get doc url init behavior."""
  fw: typing.Any = TorchAdapter()
  assert "nn.init.html" in fw.get_doc_url("torch.nn.init.constant_")


def test_torch_convert_numpy_exception(monkeypatch: typing.Any) -> None:
  """Verify torch convert numpy exception behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw: typing.Any = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch: typing.Any = MagicMock()
  mock_tensor_cls: typing.Any = type("Tensor", (), {})
  mock_torch.Tensor = mock_tensor_cls

  def mock_from_numpy(data: typing.Any) -> None:
    """Execute mock from numpy helper logic.

    Args:
        data: Test argument or fixture.
    """
    raise ValueError("mock error")

  mock_torch.from_numpy = mock_from_numpy
  mock_torch.tensor.return_value = mock_tensor_cls()
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res: typing.Any = fw.convert(np.array([1, 2]))
  assert isinstance(res, mock_tensor_cls)


def test_torch_convert_list_exception(monkeypatch: typing.Any) -> None:
  """Verify torch convert list exception behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw: typing.Any = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch: typing.Any = MagicMock()

  def mock_tensor(data: typing.Any) -> None:
    """Execute mock tensor helper logic.

    Args:
        data: Test argument or fixture.
    """
    raise ValueError("mock error")

  mock_torch.tensor = mock_tensor
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res: typing.Any = fw.convert([1, 2, 3])
  # Should fall through and return the list
  assert res == [1, 2, 3]


def test_torch_collect_ghost_no_snapshot() -> None:
  """Verify torch collect ghost no snapshot behavior."""
  fw: typing.Any = TorchAdapter()
  fw._snapshot_data = None
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  assert fw._collect_ghost(SemanticTier.LAYER) == []

  fw._snapshot_data = {"categories": {"layer": [{"api_path": "test", "name": "Test", "kind": "function"}]}}
  res: typing.Any = fw._collect_ghost(SemanticTier.LAYER)
  assert len(res) == 1
  assert res[0].api_path == "test"


def test_torch_collect_live_layer() -> typing.Any:
  """Verify torch collect live layer behavior.

  Returns:
      Computed test result or fixture value.
  """
  fw: typing.Any = TorchAdapter()
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  def dummy_scan_layers() -> typing.Any:
    """Execute dummy scan layers helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return ["layer1"]

  fw._scan_layers = dummy_scan_layers
  res: typing.Any = fw._collect_live(SemanticTier.LAYER)
  assert res == ["layer1"]


def test_torch_import_fail_in_convert(monkeypatch: typing.Any) -> typing.Any:
  """Verify torch import fail in convert behavior.

  Args:
      monkeypatch: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  import builtins

  fw: typing.Any = TorchAdapter()
  original_import: typing.Any = builtins.__import__

  def mock_import(name: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Execute mock import helper logic.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "torch":
      raise ImportError("mock error")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res: typing.Any = fw.convert([1, 2, 3])
  assert res == [1, 2, 3]


def test_torch_get_device_syntax() -> None:
  """Verify torch get device syntax behavior."""
  fw: typing.Any = TorchAdapter()
  # Hit 221
  res: typing.Any = fw.get_device_syntax("xpu")
  assert "xpu" in res

  res = fw.get_device_syntax("cuda", "1")
  assert "1" in res


def test_torch_convert_numpy_fallback(monkeypatch: typing.Any) -> typing.Any:
  """Verify torch convert numpy fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  fw: typing.Any = TorchAdapter()
  import sys

  class MockTorch:
    """Test helper class MockTorch."""

    def from_numpy(self, data: typing.Any) -> None:
      """Execute from numpy helper logic.

      Args:
          data: Test argument or fixture.
      """
      raise ValueError()

    def tensor(self, data: typing.Any) -> typing.Any:
      """Execute tensor helper logic.

      Args:
          data: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "fallback"

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res: typing.Any = fw.convert(np.array([1, 2]))
  assert res == "fallback"


def test_torch_convert_list_fallback(monkeypatch: typing.Any) -> None:
  """Verify torch convert list fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw: typing.Any = TorchAdapter()
  import sys

  class MockTorch:
    """Test helper class MockTorch."""

    def tensor(self, data: typing.Any) -> None:
      """Execute tensor helper logic.

      Args:
          data: Test argument or fixture.
      """
      raise ValueError()

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res: typing.Any = fw.convert([1, 2])
  assert res == [1, 2]


def test_torch_definitions_fallback(monkeypatch: typing.Any) -> None:
  """Verify torch definitions fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.torch as torch_mod

  monkeypatch.setattr(torch_mod, "load_definitions", lambda fw: {})
  fw: typing.Any = TorchAdapter()
  if callable(fw.definitions):
    defs: typing.Any = fw.definitions()
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

  mock_torch: typing.Any = MagicMock()
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
    adapter: typing.Any = mod.TorchAdapter()
    assert adapter._mode == mod.InitMode.LIVE
  importlib.reload(mod)


def test_torch_properties_and_methods() -> None:
  """Test properties and remaining methods on TorchAdapter."""
  from unittest.mock import MagicMock, patch

  from ml_switcheroo_ir.schema.ghost import SemanticTier

  adapter: typing.Any = TorchAdapter()
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

  s: typing.Any = {}
  adapter.apply_wiring(s)
  assert "mappings" in s

  assert "generated/torch.add.html" in (adapter.get_doc_url("torch.add") or "")

  all_defs: typing.Any = {
    "ReLU": MagicMock(),
    "Linear": MagicMock(),
    "Conv2d": MagicMock(),
    "Conv1d": MagicMock(),
    "Conv3d": MagicMock(),
    "ConvTranspose2d": MagicMock(),
  }
  with patch("ml_switcheroo.frameworks.torch.load_definitions", return_value=all_defs):
    defs: typing.Any = adapter.definitions
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
      adapter_empty: typing.Any = TorchAdapter()
      assert adapter_empty._mode == InitMode.GHOST
      assert adapter_empty._snapshot_data == {}

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={"categories": {}}):
      adapter_snap: typing.Any = TorchAdapter()
      assert adapter_snap._mode == InitMode.GHOST
      assert "categories" in adapter_snap._snapshot_data
