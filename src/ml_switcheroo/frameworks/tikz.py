"""TikZ Framework Adapter.

Simplified to only provide Metadata.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StandardMap,
  StandardCategory,
  ImportConfig,
  InitMode,
  OperationDef,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits


@register_framework("tikz")
class TikzAdapter(FrameworkAdapter):
  """Adapter for TikZ."""

  display_name: str = "TikZ (LaTeX)"
  inherits_from: Optional[str] = None
  ui_priority: int = 1000
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """Execute implementation detail."""
    pass

  @property
  def search_modules(self) -> List[str]:
    """Execute implementation detail."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Execute implementation detail."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Execute implementation detail."""
    return ("tikz", "tikz")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Execute implementation detail."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Execute implementation detail."""
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Execute implementation detail."""
    return {
      "import": "% latex package imports here",
      "convert_input": "% input {np_var}",
      "to_numpy": "% output {res_var}",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Execute implementation detail."""
    return []

  def get_harness_init_code(self) -> str:
    """Execute implementation detail."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Execute implementation detail."""
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Execute implementation detail."""
    return [SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """Execute implementation detail."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Execute implementation detail."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Execute implementation detail."""
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Execute implementation detail."""
    return load_definitions("tikz")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Execute implementation detail."""
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Execute implementation detail."""
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """Execute implementation detail."""
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Execute implementation detail."""
    return ""

  def get_device_check_syntax(self) -> str:
    """Execute implementation detail."""
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Execute implementation detail."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Execute implementation detail."""
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Execute implementation detail."""
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Execute implementation detail."""
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Execute implementation detail."""
    return "# Weights not supported in TikZ mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Execute implementation detail."""
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Execute implementation detail."""
    return "# Weights not supported in TikZ mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Execute implementation detail."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Execute implementation detail."""
    return None

  def convert(self, data: Any) -> Any:
    """Execute implementation detail."""
    return str(data)

  def get_tiered_examples(self) -> Dict[str, str]:
    """Execute implementation detail."""
    return {
      "tier1_math": "% Math ops not visualized directly",
      "tier2_neural": r"\begin{tikzpicture}\node (input) {Input}; \node (layer) [right of=input] {Layer}; \draw[->] (input) -- (layer);\end{tikzpicture}",
      "tier3_extras": "% Extras ignored",
    }
