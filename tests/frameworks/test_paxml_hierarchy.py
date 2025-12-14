"""
Tests for PaxML Adapter Hierarchy integration.

Verifies that:
1. PaxmlAdapter inherits correctly from JAXStackMixin via composition or mixin.
2. It provides specific structural traits (setup vs init).
3. It wires up Level 1 libraries (Optax/Orbax) despite being a Level 2 framework
   distinct from Flax.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


def test_paxml_inherits_stack_wiring():
  """
  Verify Level 0/1 wiring (Math & Optax) appears in PaxML snapshot.
  This confirms JAXStackMixin._apply_stack_wiring is called.
  """
  adapter = PaxmlAdapter()
  snapshot = {}

  adapter.apply_wiring(snapshot)
  mappings = snapshot["mappings"]

  # 1. Math (Level 0)
  assert "Abs" in mappings
  assert mappings["Abs"]["api"] == "jnp.abs"

  # 2. Optimization (Level 1)
  assert "Adam" in mappings
  assert mappings["Adam"]["api"] == "optax.adam"
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_paxml_structural_traits():
  """
  Verify Level 2 specifics for Praxis.
  """
  adapter = PaxmlAdapter()
  traits = adapter.structural_traits

  # Base class
  assert traits.module_base == "praxis.base_layer.BaseLayer"

  # Init Method
  assert traits.init_method_name == "setup"

  # Super Init (Not required in Praxis usually)
  assert traits.requires_super_init is False


def test_serialization_syntax_reused():
  """
  Verify PaxML uses Orbax for IO (via Mixin inheritance).
  """
  adapter = PaxmlAdapter()

  # Imports
  assert "import orbax.checkpoint" in adapter.get_serialization_imports()[0]

  # Syntax
  save_code = adapter.get_serialization_syntax("save", "dir", "obj")
  assert "orbax.checkpoint" in save_code


def test_discovery_composition():
  """
  Verify `collect_api` delegates to core for optimizers.
  (This test mocks the core adapter to verify interaction).
  """
  # We patch where JaxCoreAdapter is IMPORTED in paxml.py
  with patch("ml_switcheroo.frameworks.paxml.JaxCoreAdapter") as MockCore:
    mock_core_instance = MockCore.return_value

    # Create Dummy GhostRef simulating what Core would find
    from ml_switcheroo.core.ghost import GhostRef

    dummy_ref = GhostRef(name="sgd", api_path="optax.sgd", kind="func", params=[])

    # Core collects LIVE data
    mock_core_instance.collect_api.return_value = [dummy_ref]

    adapter = PaxmlAdapter()
    # Force live mode
    adapter._mode = "live"

    from ml_switcheroo.frameworks.base import StandardCategory

    results = adapter.collect_api(StandardCategory.OPTIMIZER)

    # Should contain the result from core
    assert len(results) == 1
    assert results[0].name == "sgd"

    # Verify call arguments
    mock_core_instance.collect_api.assert_called_with(StandardCategory.OPTIMIZER)
