"""Unit tests for `MLXAdapter` and `MlxIOMixin` (`ml_switcheroo.frameworks.mlx`)."""

import importlib
import importlib.util
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.mlx as mlx_mod
from ml_switcheroo.frameworks.base import FrameworkSnapshot, InitMode, StandardMap
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.mlx_io import MlxIOMixin


def test_mlx_adapter_metadata_traits_syntax_and_io() -> None:
  """Verify `MLXAdapter` metadata, traits, device/RNG/serialization/weight syntax, docs, and `MlxIOMixin`."""
  adapter = MLXAdapter()
  assert adapter.import_alias == ("mlx.core", "mx")

  ns = adapter.import_namespaces
  assert ns["mlx.core"].recommended_alias == "mx"
  assert ns["mlx.nn"].recommended_alias == "nn"
  assert ns["mlx.optimizers"].recommended_alias == "optim"
  assert SemanticTier.NEURAL in adapter.supported_tiers

  cfg = adapter.test_config
  assert "import mlx.core as mx" in cfg["import"]
  assert "array" in cfg["convert_input"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "tolist" in adapter.get_to_numpy_code()

  s_traits = adapter.structural_traits
  assert s_traits.module_base == "mlx.nn.Module"
  assert "__call__" in s_traits.known_inference_methods

  p_traits = adapter.plugin_traits
  assert p_traits.has_numpy_compatible_arrays is True
  assert p_traits.requires_explicit_rng is False
  assert p_traits.requires_functional_state is False
  assert adapter.rng_seed_methods == ["seed", "random.seed"]
  assert adapter.declared_magic_args == []
  assert isinstance(adapter.definitions, dict)

  with patch.object(
    mlx_mod,
    "load_definitions",
    return_value={"Linear": StandardMap(api="mlx.nn.Linear"), "Conv2d": StandardMap(api="mlx.nn.Conv2d")},
  ):
    assert "Linear" in adapter.definitions

  assert adapter.get_device_syntax("gpu", "0") == "mx.Device(mx.gpu, 0)"
  assert adapter.get_device_syntax("gpu") == "mx.Device(mx.gpu)"
  assert adapter.get_device_syntax("cpu", None) == "mx.Device(mx.cpu)"
  assert adapter.get_device_check_syntax() == "mx.default_device() == mx.gpu"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  assert MlxIOMixin().get_serialization_imports() == ["import mlx.core as mx"]
  assert adapter.get_serialization_imports() == ["import mlx.core as mx"]
  assert len(adapter.get_serialization_syntax("load", "path", "obj")) > 0
  assert len(adapter.get_serialization_syntax("load", "path")) > 0
  assert len(adapter.get_serialization_syntax("save", "path", "obj")) > 0
  assert adapter.get_serialization_syntax("other", "path") == ""

  assert len(adapter.get_weight_conversion_imports()) > 0
  assert len(adapter.get_weight_load_code("path")) > 0
  assert len(adapter.get_tensor_to_numpy_expr("t")) > 0
  assert len(adapter.get_weight_save_code("s", "path")) > 0

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}

  url_core = adapter.get_doc_url("mlx.core.abs")
  assert url_core is not None
  assert "ml-explore.github.io" in url_core
  assert adapter.get_doc_url("mlx.nn.Linear") is not None
  url_unk = adapter.get_doc_url("unknown")
  assert url_unk is not None
  assert "unknown.html" in url_unk
  assert len(adapter.get_tiered_examples()) > 0


def test_mlx_adapter_convert_and_ghost_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `MLXAdapter.convert`, `find_spec` error fallback, and missing `numpy` reload.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = MLXAdapter()

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "mlx", None)
    mp.setitem(sys.modules, "mlx.core", None)
    assert adapter.convert([1, 2]) == [1, 2]

  mock_mlx = MagicMock()
  mock_core = MagicMock()
  mock_mlx.core = mock_core
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "mlx", mock_mlx)
    mp.setitem(sys.modules, "mlx.core", mock_core)
    mock_core.array.return_value = "mlx_tensor"
    assert adapter.convert([1, 2]) == "mlx_tensor"
    assert adapter.convert(np.array([1, 2])) == "mlx_tensor"
    assert adapter.convert(1) == 1

    mock_core.array.side_effect = RuntimeError("err")
    assert adapter.convert([1, 2]) == [1, 2]

  # `find_spec` raising `ValueError` falls back to GHOST mode cleanly
  monkeypatch.delitem(sys.modules, "mlx", raising=False)
  with (
    patch.object(importlib.util, "find_spec", side_effect=ValueError("bad spec")),
    patch.object(mlx_mod, "load_snapshot_for_adapter", return_value={"version": "0.18"}),
  ):
    ghost_adapter = MLXAdapter()
    assert ghost_adapter._mode == InitMode.GHOST
    assert ghost_adapter._snapshot_data == FrameworkSnapshot(version="0.18")

  # Missing `numpy` at module import time sets `np = None`
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "numpy", None)
    importlib.reload(mlx_mod)
    assert mlx_mod.np is None
  importlib.reload(mlx_mod)
