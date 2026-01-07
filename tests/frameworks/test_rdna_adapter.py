"""
Tests for AMD RDNA Framework Adapter.

Verifies:
1. Registration in `_ADAPTER_REGISTRY`.
2. Protocol compliance (Ghost mode, empty defaults).
3. Correct loading of Math definitions (v_add_f32).
4. Engine Hooks (create_parser, create_emitter, target_arch configuration).
"""

import libcst as cst
from ml_switcheroo.frameworks.rdna import (
  RdnaAdapter,
  PythonToRdnaEmitter,
  RdnaToPythonParser,
)
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
  # Verify default architecture updated to gfx1030
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
  code = RdnaAdapter.get_example_code()
  assert "v_add_f32" in code


def test_emitter_creation_with_arch() -> None:
  """Verify factory returns valid emitter instance with arch config."""
  adapter = RdnaAdapter(target_arch="gfx1100")
  assert adapter.target_arch == "gfx1100"

  emitter = adapter.create_emitter()
  assert isinstance(emitter, PythonToRdnaEmitter)
  assert hasattr(emitter, "emit")
  assert emitter.target_arch == "gfx1100"

  # Verify basic emit behavior using a functional call to generate a graph
  # "x=1" does not produce graph nodes for current GraphExtractor.
  # Use "y = torch.abs(x)" assuming GraphExtractor can parse the call.
  # Note: Requires semantics to be partially working or fallback to unmapped but
  # valid node generation.
  out = emitter.emit("y = torch.abs(x)")
  assert "; RDNA Code Generation Initialized" in out
  assert "Arch: gfx1100" in out


def test_parser_creation() -> None:
  """Verify factory returns valid parser instance."""
  adapter = RdnaAdapter()
  code = "v_mov_b32 v0, s0"
  parser = adapter.create_parser(code)
  assert isinstance(parser, RdnaToPythonParser)

  # Basic parse check
  tree = parser.parse()
  assert isinstance(tree, cst.Module)


def test_no_op_methods() -> None:
  """Verify safety of protocol implementation."""
  adapter = RdnaAdapter()
  assert adapter.get_device_syntax("gpu") == "; Target Device: gpu"
  assert adapter.get_rng_split_syntax("r", "k") == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
