"""
Tests for NVIDIA SASS Framework Adapter and Backend Wiring.

Verifies:
1. Registration in `_ADAPTER_REGISTRY`.
2. Protocol compliance (Ghost mode, empty defaults).
3. Correct loading of Math definitions (FADD).
4. Backend connectivity (SassBackend).
"""

from unittest.mock import MagicMock
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.compiler.backends.sass import SassBackend
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
  SemanticTier,
)


def test_sass_adapter_registration():
  """Verify SASS is registered correctly in the global registry."""
  assert "sass" in _ADAPTER_REGISTRY
  adapter = get_adapter("sass")
  assert isinstance(adapter, SassAdapter)
  assert adapter.display_name == "NVIDIA SASS"


def test_initialization_defaults():
  """Verify Ghost mode init and empty search paths."""
  adapter = SassAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.search_modules == []
  assert adapter.import_alias == ("sass", "asm")


def test_definitions_loaded_correctly():
  """
  Verify that the JSON mappings are correctly loaded by the loader utility
  and associated with the adapter.
  """
  adapter = SassAdapter()
  defs = adapter.definitions

  # Check mapping exists
  assert "Add" in defs
  # Check mnemonics from generated/pre-seeded JSON
  assert defs["Add"].api == "FADD"
  assert defs["Mul"].api == "FMUL"

  # Check special function
  assert "FusedMultiplyAdd" in defs
  assert defs["FusedMultiplyAdd"].api == "FFMA32I"


def test_supported_tiers():
  """Verify supported tiers list."""
  adapter = SassAdapter()
  tiers = adapter.supported_tiers
  ## Needs to be a list containing SemanticTier enums
  assert SemanticTier.ARRAY_API in tiers
  assert len(tiers) == 1


def test_example_code():
  """Verify example code structure."""
  code = SassAdapter().get_tiered_examples()["tier1_math"]
  assert "FADD" in code
  assert "//" in code


def test_sass_backend_instantiation():
  """
  Verify that the SassBackend can be instantiated with a SemanticsManager.
  This replaces the old test for `create_emitter`.
  """
  semantics_mock = MagicMock()
  backend = SassBackend(semantics=semantics_mock)
  assert backend is not None
  # Check that it has a compile method (Protocol)
  assert hasattr(backend, "compile")
