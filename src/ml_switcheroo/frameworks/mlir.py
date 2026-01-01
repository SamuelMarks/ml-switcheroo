"""
MLIR Framework Adapter.

This module provides the pseudo-adapter for Multi-Level Intermediate Representation (MLIR).
It registers "mlir" as a valid source/target key in the system, enabling the Engine
to route conversion logic through the MLIR Emitter/Parser path instead of the
standard Python AST Rewriter path.

It does not provide semantic mappings (Definitions) because MLIR transformation
is structural, not API-based.
"""

from typing import Any, Dict, List, Set, Tuple

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  ImportConfig,
  PluginTraits,
  StandardMap,
  StructuralTraits,
  register_framework,
)
from ml_switcheroo.semantics.schema import OpDefinition


@register_framework("mlir")
class MlirAdapter:
  """
  Adapter for the Multi-Level Intermediate Representation key.

  Acts as a bridge for the ASTEngine to trigger Emitter/Generator logic.
  Unlike standard frameworks, this adapter does not scan modules or provide
  semantic mappings, as MLIR is an internal structural representation.
  """

  display_name: str = "MLIR (Intermediate)"
  # Place at the end of the list visually
  ui_priority: int = 900
  inherits_from = None

  def __init__(self) -> None:
    """Initialize the adapter."""
    pass

  @property
  def search_modules(self) -> List[str]:
    """MLIR is not a Python package, so no modules to scan."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """No unsafe submodules."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Virtual alias for import logic."""
    return ("mlir", "sw")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """No namespaces to map."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """No heuristics needed."""
    return {}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """MLIR supports representing all tiers structurally."""
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def structural_traits(self) -> StructuralTraits:
    """Default traits (no special class rewriting rules)."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Default traits."""
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    No semantic definitions.

    Logic is handled by the Emitter/Generator, not the semantic rewriter.
    """
    return {}

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """No semantic specifications."""
    return {}

  # Required methods for Protocol compliance
  @property
  def harness_imports(self) -> List[str]:
    """No harness imports."""
    return []

  @property
  def test_config(self) -> Dict[str, str]:
    """No test config."""
    return {}

  def get_harness_init_code(self) -> str:
    """No initialization code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """No runtime conversion logic."""
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> List[str]:
    """No magic args."""
    return []

  def convert(self, data: Any) -> Any:
    """Identity conversion for testing."""
    return str(data)

  def get_device_syntax(self, device_type: str, device_index: str = None) -> str:
    """No device syntax."""
    return ""

  def get_device_check_syntax(self) -> str:
    """No device checks."""
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """No RNG splitting."""
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """No serialization imports."""
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: str = None) -> str:
    """No serialization syntax."""
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No manual wiring."""
    pass

  @property
  def rng_seed_methods(self) -> List[str]:
    """No seed methods."""
    return []

  @classmethod
  def get_example_code(cls) -> str:
    """Return a placeholder description."""
    return "# MLIR Intermediate Representation\n# Used for verifying structural integrity."

  def get_tiered_examples(self) -> Dict[str, str]:
    """Return placeholders."""
    code = self.get_example_code()
    return {
      "tier1_math": code,
      "tier2_neural": code,
      "tier3_extras": code,
    }
