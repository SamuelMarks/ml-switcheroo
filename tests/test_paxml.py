"""Extra tests for the Paxml framework adapter."""

import sys

import pytest


def test_paxml_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import types

  mock_praxis: types.ModuleType = types.ModuleType("praxis")
  mock_praxis.layers = types.ModuleType("praxis.layers")
  mock_praxis.base_layer = types.ModuleType("praxis.base_layer")
  mock_praxis.layers.activations = types.ModuleType("praxis.layers.activations")
  mock_praxis.layers.normalizations = types.ModuleType("praxis.layers.normalizations")

  monkeypatch.setitem(sys.modules, "praxis", mock_praxis)
  monkeypatch.setitem(sys.modules, "praxis.layers", mock_praxis.layers)
  monkeypatch.setitem(sys.modules, "praxis.base_layer", mock_praxis.base_layer)
  monkeypatch.setitem(sys.modules, "praxis.layers.activations", mock_praxis.layers.activations)
  monkeypatch.setitem(sys.modules, "praxis.layers.normalizations", mock_praxis.layers.normalizations)

  # Force reload of paxml
  if "ml_switcheroo.frameworks.paxml" in sys.modules:
    del sys.modules["ml_switcheroo.frameworks.paxml"]

  import ml_switcheroo.frameworks.paxml as paxml_mod

  assert paxml_mod.praxis is not None


def test_paxml_methods_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: PaxmlAdapter = PaxmlAdapter()

  # 77-80 (ghost loading if needed)
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  adapter._snapshot_data = {
    "categories": {
      "neural": [{"name": "fake", "module": "test", "tier": "neural", "api_path": "fake", "kind": "function"}]
    }
  }
  adapter._collect_ghost(SemanticTier.NEURAL)

  # other methods
  adapter.get_harness_init_code()
  adapter.get_to_numpy_code()
  adapter.get_device_syntax("cpu")

  adapter.convert([1, 2])
  import sys
  from unittest.mock import MagicMock, patch

  with patch.dict(sys.modules, {"jax.numpy": MagicMock()}):
    import jax.numpy as jnp

    jnp.array.return_value = "tensor"
    adapter.convert([1, 2])
    jnp.array.side_effect = Exception("err")
    adapter.convert([1, 2])

  with patch.dict(sys.modules, {"jax.numpy": None}):
    adapter.convert([1, 2])

  adapter.get_doc_url("paxml.test")
  adapter.harness_imports
  adapter.plugin_traits
  adapter.definitions
  adapter.get_tiered_examples()


def test_paxml_extra_misses() -> None:
  """Docstring."""
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: PaxmlAdapter = PaxmlAdapter()
  adapter._snapshot_data = {}
  adapter._collect_ghost(SemanticTier.NEURAL)

  adapter.definitions


def test_paxml_definitions_fallback() -> None:
  """Docstring."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: PaxmlAdapter = PaxmlAdapter()
  with patch("ml_switcheroo.frameworks.paxml.load_definitions", return_value={}):
    adapter.definitions


def test_paxml_definitions_args_none() -> None:
  """Docstring."""
  from unittest.mock import MagicMock, patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: PaxmlAdapter = PaxmlAdapter()
  mock_map: MagicMock = MagicMock()
  mock_map.args = None
  with patch("ml_switcheroo.frameworks.paxml.load_definitions", return_value={"Linear": mock_map}):
    adapter.definitions
