"""Test suite for torch."""

import typing
from pydantic import JsonValue
import pytest


import numpy as np

from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_no_torch_module(monkeypatch: pytest.MonkeyPatch) -> None:
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
  fw = TorchAdapter()
  assert "nn.init.html" in fw.get_doc_url("torch.nn.init.constant_")  # type: ignore[operator]


def test_torch_convert_numpy_exception(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch convert numpy exception behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch = MagicMock()
  mock_tensor_cls = type("Tensor", (), {})
  mock_torch.Tensor = mock_tensor_cls

  def mock_from_numpy(data: np.ndarray[tuple[int, ...], np.dtype[np.int64]]) -> None:
    """Execute mock from numpy helper logic.

    Args:
        data: Test argument or fixture.
    """
    raise ValueError("mock error")

  mock_torch.from_numpy = mock_from_numpy
  mock_torch.tensor.return_value = mock_tensor_cls()
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res = fw.convert(np.array([1, 2]))
  assert isinstance(res, mock_tensor_cls)


def test_torch_convert_list_exception(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch convert list exception behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw = TorchAdapter()
  import sys
  from unittest.mock import MagicMock

  mock_torch = MagicMock()

  def mock_tensor(data: list[int]) -> None:
    """Execute mock tensor helper logic.

    Args:
        data: Test argument or fixture.
    """
    raise ValueError("mock error")

  mock_torch.tensor = mock_tensor
  monkeypatch.setitem(sys.modules, "torch", mock_torch)

  res = fw.convert([1, 2, 3])
  # Should fall through and return the list
  assert res == [1, 2, 3]


def test_torch_collect_ghost_no_snapshot() -> None:
  """Verify torch collect ghost no snapshot behavior."""
  fw = TorchAdapter()
  fw._snapshot_data = None  # type: ignore[assignment]
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  assert fw._collect_ghost(SemanticTier.LAYER) == []

  fw._snapshot_data = {"categories": {"layer": [{"api_path": "test", "name": "Test", "kind": "function"}]}}
  res = fw._collect_ghost(SemanticTier.LAYER)
  assert len(res) == 1
  assert res[0].api_path == "test"


def test_torch_collect_live_layer() -> None:
  """Verify torch collect live layer behavior.

  Returns:
      Computed test result or fixture value.
  """
  fw = TorchAdapter()
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  def dummy_scan_layers() -> list[str]:
    """Execute dummy scan layers helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return ["layer1"]

  fw._scan_layers = dummy_scan_layers  # type: ignore[attr-defined]
  res = fw._collect_live(SemanticTier.LAYER)
  assert res == ["layer1"]  # type: ignore[comparison-overlap]


def test_torch_import_fail_in_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch import fail in convert behavior.

  Args:
      monkeypatch: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  import builtins
  import types

  fw = TorchAdapter()
  original_import = builtins.__import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
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
    return original_import(name, *args, **kwargs)  # type: ignore[arg-type]

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res = fw.convert([1, 2, 3])
  assert res == [1, 2, 3]


def test_torch_get_device_syntax() -> None:
  """Verify torch get device syntax behavior."""
  fw = TorchAdapter()
  # Hit 221
  res = fw.get_device_syntax("xpu")
  assert "xpu" in res

  res = fw.get_device_syntax("cuda", "1")
  assert "1" in res


def test_torch_convert_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch convert numpy fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Test helper class MockTorch."""

    def from_numpy(self, data: np.ndarray[tuple[int, ...], np.dtype[np.int64]]) -> None:
      """Execute from numpy helper logic.

      Args:
          data: Test argument or fixture.
      """
      raise ValueError()

    def tensor(self, data: np.ndarray[tuple[int, ...], np.dtype[np.int64]]) -> str:
      """Execute tensor helper logic.

      Args:
          data: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "fallback"

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert(np.array([1, 2]))
  assert res == "fallback"


def test_torch_convert_list_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch convert list fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  fw = TorchAdapter()
  import sys

  class MockTorch:
    """Test helper class MockTorch."""

    def tensor(self, data: list[int]) -> None:
      """Execute tensor helper logic.

      Args:
          data: Test argument or fixture.
      """
      raise ValueError()

  monkeypatch.setitem(sys.modules, "torch", MockTorch())
  res = fw.convert([1, 2])
  assert res == [1, 2]


def test_torch_definitions_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify torch definitions fallback behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
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

  adapter = TorchAdapter()
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

  s: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  adapter.apply_wiring(s)  # type: ignore[arg-type]
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
      adapter_empty = TorchAdapter()
      assert adapter_empty._mode == InitMode.GHOST
      assert adapter_empty._snapshot_data == {}

    with patch("ml_switcheroo.frameworks.torch.load_snapshot_for_adapter", return_value={"categories": {}}):
      adapter_snap = TorchAdapter()
      assert adapter_snap._mode == InitMode.GHOST
      assert "categories" in adapter_snap._snapshot_data
