"""
AMD RDNA / GCN Framework Adapter.

Provides metadata and configuration for the RDNA compiler stack.
This adapter acts as a metadata container for the Compiler Registry,
identifying RDNA as a target language and providing static definitions.

Migration Note:
    Legacy shim classes (`PythonToRdnaEmitter`, `RdnaToPythonParser`)
    have been removed. Routing now occurs via `compiler.registry`.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StandardMap,
  StandardCategory,
  ImportConfig,
  InitMode,
  OpDefinition,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


@register_framework("rdna")
class RdnaAdapter(FrameworkAdapter):
  """
  Adapter for AMD RDNA.
  """

  display_name: str = "AMD RDNA"
  inherits_from: Optional[str] = None
  ui_priority: int = 151
  _mode: InitMode = InitMode.GHOST

  def __init__(self, target_arch: str = "gfx1030") -> None:
    """
    Initialize the adapter.

    Args:
        target_arch: Target GPU architecture (e.g. gfx1030).
                     Note: This property is primarily informational in the new architecture.
    """
    self.target_arch = target_arch

  @property
  def search_modules(self) -> List[str]:
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("rdna", "asm")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    return {
      "import": "; RDNA Header",
      "convert_input": "; Input {np_var}",
      "to_numpy": "; Output {res_var}",
    }

  @property
  def harness_imports(self) -> List[str]:
    return []

  def get_harness_init_code(self) -> str:
    return ""

  def get_to_numpy_code(self) -> str:
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    return load_definitions("rdna")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return f"; Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    return ""

  def get_serialization_imports(self) -> List[str]:
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    return "; Weights loading not supported in RDNA adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    return "; Weights saving not supported in RDNA adapter"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    return f"https://gpuopen.com/learn/rdna-performance-guide/?q={api_name}"

  def convert(self, data: Any) -> Any:
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    return "v_add_f32 v0, v1, v2"

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "; Neural layers map to comment blocks",
      "tier3_extras": "; Extras ignored",
    }
