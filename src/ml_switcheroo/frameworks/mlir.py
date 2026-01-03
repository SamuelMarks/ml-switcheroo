"""
MLIR Intermediate Representation Adapter.

This adapter provides support for pivoting through the Multi-Level Intermediate Representation
(MLIR) format. It serves primarily as a validation target for the `Python -> MLIR -> Python`
roundtrip pipeline.

It exposes NO definitions (empty dict) as MLIR translation is handled structurally
by the Emitter/Parser, not by semantic mapping.
"""

from typing import Dict, List, Optional, Any, Tuple, Set

from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
  InitMode,
  GhostRef,
)
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("mlir")
class MlirAdapter(FrameworkAdapter):
  """
  Adapter for MLIR (Intermediate).
  """

  display_name: str = "MLIR (Intermediate)"
  ui_priority: int = 2000
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """Init."""
    self._mode = InitMode.GHOST

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns empty definitions as MLIR logic is structural, not mapped.
    Still uses loader for consistency if a file is ever added.

    Returns:
        Dict[str, StandardMap]: Definitions.
    """
    return load_definitions("mlir")

  @property
  def specifications(self) -> Dict:
    """Empty specs."""
    return {}

  @property
  def search_modules(self) -> List[str]:
    """Empty."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Empty."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """("mlir", "sw")."""
    return ("mlir", "sw")

  @property
  def import_namespaces(self) -> Dict:
    """Empty."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict:
    """Empty."""
    return {}

  @property
  def test_config(self) -> Dict:
    """Empty."""
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """Empty."""
    return []

  def get_harness_init_code(self) -> str:
    """Empty."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Identity."""
    return "return data"

  def convert(self, data: Any) -> Any:
    """Identity string conversion."""
    return str(data)

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """All tiers implicitly."""
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """Empty."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Default."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Default."""
    return PluginTraits()

  @property
  def rng_seed_methods(self) -> List[str]:
    """Empty."""
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Empty."""
    return ""

  def get_device_check_syntax(self) -> str:
    """False."""
    return "False"

  def get_rng_split_syntax(self, r, k) -> str:
    """Empty."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Empty."""
    return []

  def get_serialization_syntax(self, op: str, f: str, o: Optional[str] = None) -> str:
    """Empty."""
    return ""

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """None."""
    return None

  @classmethod
  def get_example_code(cls) -> str:
    """MLIR text."""
    return '%0 = "sw.op"() { name = "AbstractOp" } : () -> i32'

  def get_tiered_examples(self) -> Dict[str, str]:
    """Tiered examples."""
    ex = self.get_example_code()
    return {"tier1_math": ex, "tier2_neural": '%model = "sw.module"() ( { ... } ) : () -> !sw.module', "tier3_extras": ex}

  def collect_api(self, c) -> List:
    """Empty."""
    return []

  def apply_wiring(self, s) -> None:
    """None."""
    pass
