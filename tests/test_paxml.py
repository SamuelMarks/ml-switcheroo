"""Extra tests for the Paxml framework adapter."""

import typing

import sys


def test_paxml_import_success(monkeypatch: typing.Any) -> None:
  """Verify paxml import success behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import types
  from typing import Any

  mock_praxis: Any = types.ModuleType("praxis")
  mock_praxis.layers = types.ModuleType("praxis.layers")
  mock_praxis.base_layer = types.ModuleType("praxis.base_layer")
  mock_praxis.layers.activations = types.ModuleType("praxis.layers.activations")
  mock_praxis.layers.normalizations = types.ModuleType("praxis.layers.normalizations")

  monkeypatch.setitem(sys.modules, "praxis", mock_praxis)
  monkeypatch.setitem(sys.modules, "praxis.layers", mock_praxis.layers)
  monkeypatch.setitem(sys.modules, "praxis.base_layer", mock_praxis.base_layer)
  monkeypatch.setitem(sys.modules, "praxis.layers.activations", mock_praxis.layers.activations)
  monkeypatch.setitem(sys.modules, "praxis.layers.normalizations", mock_praxis.layers.normalizations)

  import importlib
  import ml_switcheroo.frameworks.paxml as paxml_mod

  try:
    importlib.reload(paxml_mod)
    assert paxml_mod.praxis is not None
  finally:
    monkeypatch.undo()
    importlib.reload(paxml_mod)


def test_paxml_methods_coverage() -> None:
  """Verify paxml methods coverage behavior."""
  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: typing.Any = PaxmlAdapter()

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

  mock_jax: typing.Any = MagicMock()
  mock_jnp: typing.Any = MagicMock()
  mock_jax.numpy = mock_jnp
  with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
    mock_jnp.array.return_value = "tensor"
    adapter.convert([1, 2])
    mock_jnp.array.side_effect = RuntimeError("err")
    adapter.convert([1, 2])

  with patch.dict(sys.modules, {"jax.numpy": None}):
    adapter.convert([1, 2])

  adapter.get_doc_url("paxml.test")
  adapter.harness_imports
  adapter.plugin_traits
  adapter.definitions
  _res: typing.Any = adapter.get_tiered_examples()
  assert isinstance(_res, dict)
  assert len(_res) > 0


def test_paxml_extra_misses() -> None:
  """Verify paxml extra misses behavior."""
  from ml_switcheroo_ir.schema.ghost import SemanticTier

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: typing.Any = PaxmlAdapter()
  adapter._snapshot_data = {}
  ghost_res: typing.Any = adapter._collect_ghost(SemanticTier.NEURAL)

  assert ghost_res == []
  assert isinstance(adapter.definitions, dict)


def test_paxml_definitions_fallback() -> None:
  """Verify paxml definitions fallback behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: typing.Any = PaxmlAdapter()
  with patch("ml_switcheroo.frameworks.paxml.load_definitions", return_value={}):
    defs: typing.Any = adapter.definitions
  assert "Linear" in defs


def test_paxml_definitions_args_none() -> None:
  """Verify paxml definitions args none behavior."""
  from unittest.mock import MagicMock, patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  adapter: typing.Any = PaxmlAdapter()
  mock_map: typing.Any = MagicMock()
  mock_map.args = None
  with patch("ml_switcheroo.frameworks.paxml.load_definitions", return_value={"Linear": mock_map}):
    defs: typing.Any = adapter.definitions
  assert "Linear" in defs


def test_paxml_init_no_snapshot() -> None:
  """Verify paxml init no snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  with (
    patch("ml_switcheroo.frameworks.paxml.praxis", None),
    patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value={}),
  ):
    adapter: typing.Any = PaxmlAdapter()
    assert adapter._snapshot_data == {}


def test_paxml_init_with_snapshot() -> None:
  """Verify paxml init with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  with (
    patch("ml_switcheroo.frameworks.paxml.praxis", None),
    patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value={"categories": {}}),
  ):
    adapter: typing.Any = PaxmlAdapter()
    assert adapter._mode.name == "GHOST"
