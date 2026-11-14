"""Unit tests for `TorchAdapter` (`ml_switcheroo.frameworks.torch`)."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.torch as torch_mod
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import FrameworkSnapshot, InitMode, StandardMap
from ml_switcheroo.frameworks.torch import TorchAdapter


def test_torch_adapter_metadata_traits_syntax_and_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `TorchAdapter` metadata, namespaces, traits, syntax helpers, doc URLs, and fallback definitions.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = TorchAdapter()
  assert adapter.import_alias == ("torch", "torch")
  ns = adapter.import_namespaces
  assert ns["torch"].recommended_alias == "torch"
  assert ns["torch.nn"].recommended_alias == "nn"
  assert ns["torch.nn.functional"].recommended_alias == "F"
  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg = adapter.test_config
  assert "import torch" in cfg["import"]
  assert "tensor" in cfg["convert_input"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "detach().cpu().numpy()" in adapter.get_to_numpy_code()

  s_traits = adapter.structural_traits
  assert s_traits.module_base == "torch.nn.Module"
  assert s_traits.forward_method == "forward"

  p_traits = adapter.plugin_traits
  assert p_traits.has_numpy_compatible_arrays is False
  assert p_traits.requires_explicit_rng is False
  assert p_traits.requires_functional_state is False
  assert adapter.rng_seed_methods == ["manual_seed", "seed"]
  assert adapter.declared_magic_args == []

  assert adapter.get_device_syntax("cuda", "0") == "torch.device(cuda, 0)"
  assert adapter.get_device_syntax("cpu", None) == "torch.device(cpu)"
  assert "xpu" in adapter.get_device_syntax("xpu")
  assert adapter.get_device_check_syntax() == "torch.cuda.is_available()"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  assert "nn.init.html" in (adapter.get_doc_url("torch.nn.init.constant_") or "")
  assert "generated/torch.add.html" in (adapter.get_doc_url("torch.add") or "")
  assert "pytorch.org" in (adapter.get_doc_url("torch.nn.Linear") or "")
  assert "pytorch.org" in (adapter.get_doc_url("unknown") or "")
  assert "tier1_math" in adapter.get_tiered_examples()

  snap_empty: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snap_empty)
  assert "mappings" in snap_empty

  snap_existing: dict[str, dict[str, JsonValue]] = {"mappings": {}}
  adapter.apply_wiring(snap_existing)
  assert snap_existing == {"mappings": {}}

  # `definitions` fallback population vs pre-populated definitions
  monkeypatch.setattr(torch_mod, "load_definitions", lambda _fw: {})
  fallback_defs = adapter.definitions
  assert {"ReLU", "Linear", "Conv2d"} <= set(fallback_defs.keys())

  prepopulated = {
    k: StandardMap(api=f"custom.{k}") for k in ("ReLU", "Linear", "Conv2d", "Conv1d", "Conv3d", "ConvTranspose2d")
  }
  monkeypatch.setattr(torch_mod, "load_definitions", lambda _fw: dict(prepopulated))
  custom_defs = adapter.definitions
  assert custom_defs["Conv2d"].api == "custom.Conv2d"


def test_torch_adapter_modes_collection_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify LIVE/GHOST initialization, `_collect_ghost`/`_collect_live`, module reload, and `convert` fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  # GHOST initialization with and without snapshot
  with patch.object(torch_mod, "torch", None):
    with patch.object(torch_mod, "load_snapshot_for_adapter", return_value={}):
      ghost_empty = TorchAdapter()
      assert ghost_empty._mode == InitMode.GHOST
      assert ghost_empty._snapshot_data == FrameworkSnapshot()
      ghost_empty._snapshot_data = None
      assert ghost_empty._collect_ghost(SemanticTier.LAYER) == []

    with patch.object(
      torch_mod,
      "load_snapshot_for_adapter",
      return_value={
        "version": "2.4",
        "categories": {"layer": [{"api_path": "torch.nn.Linear", "name": "Linear", "kind": "class"}]},
      },
    ):
      ghost_snap = TorchAdapter()
      assert ghost_snap._mode == InitMode.GHOST
      assert ghost_snap._snapshot_data is not None
      assert ghost_snap._snapshot_data.version == "2.4"
      layers = ghost_snap._collect_ghost(SemanticTier.LAYER)
      assert len(layers) == 1
      assert layers[0].api_path == "torch.nn.Linear"

  # `_collect_live` across tiers
  adapter = TorchAdapter()
  assert adapter._collect_live(SemanticTier.LOSS) == []
  assert adapter._collect_live(SemanticTier.OPTIMIZER) == []
  assert adapter._collect_live(SemanticTier.ACTIVATION) == []
  assert adapter._collect_live(SemanticTier.LAYER) == []
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []

  layer_ref = GhostRef(name="Linear", api_path="torch.nn.Linear", kind="class")
  monkeypatch.setattr(adapter, "_scan_layers", lambda: [layer_ref], raising=False)
  assert adapter._collect_live(SemanticTier.LAYER) == [layer_ref]

  # `convert`: live/mocked torch, `from_numpy` fallback to `tensor()`, `tensor()` failure passthrough, and ImportError
  assert adapter.convert(42) == 42

  class _MockTorchHappy:
    """Stub `torch` module succeeding on `from_numpy` and `tensor`."""

    @staticmethod
    def from_numpy(_arr: np.ndarray[tuple[int, ...], np.dtype[np.int64]]) -> str:
      """Return `'from_np'`.

      Args:
          _arr: NumPy array.

      Returns:
          Literal `'from_np'`.
      """
      return "from_np"

    @staticmethod
    def tensor(_data: list[int]) -> str:
      """Return `'from_list'`.

      Args:
          _data: Python list.

      Returns:
          Literal `'from_list'`.
      """
      return "from_list"

  class _MockTorchFallback:
    """Stub `torch` module raising `ValueError` on `from_numpy` and falling back to `tensor`."""

    @staticmethod
    def from_numpy(_arr: np.ndarray[tuple[int, ...], np.dtype[np.int64]]) -> str:
      """Raise `ValueError`.

      Args:
          _arr: NumPy array.

      Returns:
          Never returns; raises `ValueError`.

      Raises:
          ValueError: Always raised.
      """
      raise ValueError("from_numpy failed")

    @staticmethod
    def tensor(data: np.ndarray[tuple[int, ...], np.dtype[np.int64]] | list[int]) -> str:
      """Return `'fallback_tensor'` for ndarray and raise `ValueError` for list.

      Args:
          data: Input data.

      Returns:
          Literal `'fallback_tensor'`.

      Raises:
          ValueError: When `data` is a `list`.
      """
      if isinstance(data, list):
        raise ValueError("list failed")
      return "fallback_tensor"

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "torch", _MockTorchHappy())
    assert adapter.convert(np.array([1, 2])) == "from_np"
    assert adapter.convert([1, 2]) == "from_list"
    assert adapter.convert(42) == 42

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "torch", _MockTorchFallback())
    assert adapter.convert(np.array([1, 2])) == "fallback_tensor"
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "torch", None)
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]
    importlib.reload(torch_mod)
    assert torch_mod.torch is None

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "torch", MagicMock())
    mp.setitem(sys.modules, "torch.nn", MagicMock())
    mp.setitem(sys.modules, "torch.nn.functional", MagicMock())
    mp.setitem(sys.modules, "torch.optim", MagicMock())
    importlib.reload(torch_mod)
    assert TorchAdapter()._mode == InitMode.LIVE

  importlib.reload(torch_mod)
