"""
Tests for AMD RDNA Framework Adapter and Backend Wiring.

Verifies:
1. Registration in `_ADAPTER_REGISTRY`.
2. Protocol compliance (Ghost mode, empty defaults).
3. Correct loading of Math definitions (v_add_f32).
4. Backend connectivity (RdnaBackend).
"""

from unittest.mock import MagicMock
from ml_switcheroo.frameworks.rdna import RdnaAdapter
from ml_switcheroo.compiler.backends.rdna import RdnaBackend

from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
  SemanticTier,
)


def test_rdna_adapter_registration() -> None:
  """Verify RDNA is registered correctly in the global registry."""
  assert "rdna" in _ADAPTER_REGISTRY
  adapter = get_adapter("rdna")
  assert isinstance(adapter, RdnaAdapter)
  assert adapter.display_name == "AMD RDNA"


def test_initialization_defaults() -> None:
  """Verify Ghost mode init and empty search paths."""
  adapter = RdnaAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.search_modules == []
  assert adapter.import_alias == ("rdna", "asm")
  # Verify default architecture
  assert adapter.target_arch == "gfx1030"


def test_definitions_loaded_from_json() -> None:
  """
  Verify that the JSON mappings are correctly loaded by the loader utility
  and associated with the adapter.
  """
  adapter = RdnaAdapter()
  defs = adapter.definitions

  # Check mapping exists
  assert "Add" in defs
  # Check mnemonics from rdna.json
  assert defs["Add"].api == "v_add_f32"
  assert defs["Mul"].api == "v_mul_f32"
  assert defs["FusedMultiplyAdd"].api == "v_fmac_f32"
  # Check simple move logic
  assert defs["Move"].api == "v_mov_b32"


def test_supported_tiers() -> None:
  """Verify supported tiers list."""
  adapter = RdnaAdapter()
  tiers = adapter.supported_tiers
  ## Needs to be a list containing SemanticTier enums
  assert SemanticTier.ARRAY_API in tiers
  assert len(tiers) == 1


def test_example_code() -> None:
  """Verify example code structure."""
  code = RdnaAdapter().get_tiered_examples()["tier1_math"]
  assert "v_add_f32" in code


def test_rdna_backend_instantiation() -> None:
  """
  Verify that RdnaBackend can be instantiated with a SemanticsManager.
  Replaces the old test `create_emitter`.
  """
  semantics_mock = MagicMock()
  backend = RdnaBackend(semantics=semantics_mock)
  assert backend is not None
  assert hasattr(backend, "compile")
  # Check default architecture in backend
  assert backend.target_arch == "gfx1030"


def test_no_op_methods() -> None:
  """Verify safety of protocol implementation."""
  adapter = RdnaAdapter()
  assert adapter.get_device_syntax("gpu") == "; Target Device: gpu"
  assert adapter.get_rng_split_syntax("r", "k") == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
