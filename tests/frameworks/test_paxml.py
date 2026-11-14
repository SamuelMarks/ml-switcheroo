"""Unit tests for `PaxmlAdapter` (`ml_switcheroo.frameworks.paxml`)."""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.paxml as pax_mod
from ml_switcheroo.frameworks.base import FrameworkSnapshot, InitMode, StandardMap
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


def test_paxml_adapter_metadata_traits_definitions_and_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `PaxmlAdapter` metadata, traits, definitions, device syntax, wiring, and doc URLs.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = PaxmlAdapter()
  assert adapter.display_name == "PaxML / Praxis"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 60
  assert adapter.import_alias == ("praxis.layers", "pl")

  ns = adapter.import_namespaces
  assert {"praxis.layers", "praxis.base_layer"} <= set(ns.keys())
  assert "import praxis.layers as pl" in adapter.test_config["import"]
  assert "import jax" in adapter.harness_imports
  assert "def _make_jax_key(seed):" in adapter.get_harness_init_code()
  assert len(adapter.get_to_numpy_code()) > 0
  assert {SemanticTier.ARRAY_API, SemanticTier.NEURAL} <= set(adapter.supported_tiers)
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == []

  s_traits = adapter.structural_traits
  assert s_traits.module_base == "praxis.base_layer.BaseLayer"
  assert s_traits.init_method_name == "setup"
  assert s_traits.forward_method == "__call__"
  assert s_traits.requires_super_init is False

  p_traits = adapter.plugin_traits
  assert p_traits.has_numpy_compatible_arrays is True
  assert p_traits.requires_explicit_rng is True
  assert p_traits.requires_functional_control_flow is True
  assert p_traits.enforce_purity_analysis is True

  assert "jax" in adapter.get_device_syntax("cpu")
  assert len(adapter.get_device_check_syntax()) > 0
  assert "github.com" in (adapter.get_doc_url("praxis.layers.Linear") or "")
  examples = adapter.get_tiered_examples()
  assert {"tier1_math", "tier2_neural", "tier3_extras", "tier4_qwen3-vl"} <= set(examples.keys())

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot

  # `definitions`: default, empty `load_definitions` fallback, and `Linear` with `args=None`
  defs = adapter.definitions
  lin_args = defs["Linear"].args
  assert lin_args is not None
  assert lin_args["in_features"] == "input_dims"
  assert lin_args["bias"] == "use_bias"

  monkeypatch.setattr(pax_mod, "load_definitions", lambda _fw: {})
  fallback_defs = PaxmlAdapter().definitions
  assert {"Linear", "Sequential", "ReLU"} <= set(fallback_defs.keys())

  lin_none = MagicMock()
  lin_none.args = None
  monkeypatch.setattr(
    pax_mod,
    "load_definitions",
    lambda _fw: {"Linear": lin_none, "Sequential": StandardMap(api="s"), "ReLU": StandardMap(api="r")},
  )
  repaired_defs = PaxmlAdapter().definitions
  assert repaired_defs["Linear"].args is not None


def test_paxml_adapter_modes_ghost_collection_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify LIVE/GHOST initialization, `_collect_ghost`, module reload, and `convert` delegation.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(pax_mod, "praxis", True)
  assert PaxmlAdapter()._mode == InitMode.LIVE

  with (
    patch.object(pax_mod, "praxis", None),
    patch.object(pax_mod, "load_snapshot_for_adapter", return_value={}),
  ):
    ghost_empty = PaxmlAdapter()
    assert ghost_empty._mode == InitMode.GHOST
    assert ghost_empty._snapshot_data == FrameworkSnapshot()
    ghost_empty._snapshot_data = None
    assert ghost_empty._collect_ghost(SemanticTier.EXTRAS) == []

  with (
    patch.object(pax_mod, "praxis", None),
    patch.object(
      pax_mod,
      "load_snapshot_for_adapter",
      return_value={
        "version": "1.2",
        "categories": {"neural": [{"name": "Linear", "api_path": "praxis.layers.Linear", "kind": "class"}]},
      },
    ),
  ):
    ghost_snap = PaxmlAdapter()
    assert ghost_snap._snapshot_data is not None
    assert ghost_snap._snapshot_data.version == "1.2"
    assert len(ghost_snap._collect_ghost(SemanticTier.NEURAL)) == 1
    assert ghost_snap._collect_ghost(SemanticTier.EXTRAS) == []

  # `convert` delegates to `JaxCoreAdapter.convert`
  adapter = PaxmlAdapter()
  assert adapter.convert("test") == "test"
  with patch("ml_switcheroo.frameworks.paxml.JaxCoreAdapter.convert", return_value="converted"):
    assert adapter.convert([1, 2, 3]) == "converted"

  # Module-level reload with `praxis` available vs missing
  with pytest.MonkeyPatch.context() as mp:
    mock_praxis = types.ModuleType("praxis")
    mock_layers = types.ModuleType("praxis.layers")
    mock_base = types.ModuleType("praxis.base_layer")
    mock_act = types.ModuleType("praxis.layers.activations")
    mock_norm = types.ModuleType("praxis.layers.normalizations")
    mp.setattr(mock_praxis, "layers", mock_layers, raising=False)
    mp.setattr(mock_praxis, "base_layer", mock_base, raising=False)
    mp.setattr(mock_layers, "activations", mock_act, raising=False)
    mp.setattr(mock_layers, "normalizations", mock_norm, raising=False)
    mp.setitem(sys.modules, "praxis", mock_praxis)
    mp.setitem(sys.modules, "praxis.layers", mock_layers)
    mp.setitem(sys.modules, "praxis.base_layer", mock_base)
    mp.setitem(sys.modules, "praxis.layers.activations", mock_act)
    mp.setitem(sys.modules, "praxis.layers.normalizations", mock_norm)
    importlib.reload(pax_mod)
    assert PaxmlAdapter()._mode == InitMode.LIVE

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "paxml", None)
    mp.setitem(sys.modules, "praxis", None)
    importlib.reload(pax_mod)
    assert pax_mod.praxis is None

  importlib.reload(pax_mod)
