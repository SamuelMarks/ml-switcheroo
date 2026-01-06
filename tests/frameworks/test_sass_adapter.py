"""
Tests for NVIDIA SASS Framework Adapter.

Verifies:
1. Registration in `_ADAPTER_REGISTRY`.
2. Protocol compliance (Ghost mode, empty defaults).
3. Correct loading of Math definitions (FADD).
4. Example code generation.
"""

from ml_switcheroo.frameworks.sass import SassAdapter
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
  code = SassAdapter.get_example_code()
  assert "FADD" in code
  assert "//" in code


def test_no_op_methods():
  """Verify safety of protocol implementation."""
  adapter = SassAdapter()
  assert adapter.get_device_syntax("cuda") == "// Target Device: cuda"
  assert adapter.get_rng_split_syntax("r", "k") == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
