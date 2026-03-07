"""
StableHLO Framework Adapter.

Provides metadata and hooks for the MLIR/StableHLO stack.
This adapter acts as a metadata container for the Compiler Registry,
identifying StableHLO as a target language and providing static definitions.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import (
  ImportConfig,
  InitMode,
  StandardCategory,
  StandardMap,
  register_framework,
  FrameworkAdapter,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import OperationDef, PluginTraits, StructuralTraits


@register_framework("stablehlo")
class StableHloAdapter(FrameworkAdapter):
  """Adapter for StableHLO."""

  display_name: str = "StableHLO (MLIR)"
  inherits_from: Optional[str] = None
  ui_priority: int = 95
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """TODO: Add docstring."""
    pass

  @property
  def search_modules(self) -> List[str]:  # pragma: no cover
    """TODO: Add docstring."""
    return []

  @property  # pragma: no cover
  def unsafe_submodules(self) -> Set[str]:
    """TODO: Add docstring."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """TODO: Add docstring."""
    return ("stablehlo", "stablehlo")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """TODO: Add docstring."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """TODO: Add docstring."""
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """TODO: Add docstring."""
    return {
      "import": "// module attributes",
      "convert_input": "// input tensor {np_var}",
      "to_numpy": "// result tensor {res_var}",
    }

  @property
  def harness_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  def get_harness_init_code(self) -> str:
    """TODO: Add docstring."""
    return ""

  def get_to_numpy_code(self) -> str:
    """TODO: Add docstring."""
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """TODO: Add docstring."""
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]

  # pragma: no cover
  @property
  def declared_magic_args(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """TODO: Add docstring."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """TODO: Add docstring."""
    return PluginTraits()  # pragma: no cover

  # pragma: no cover
  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """TODO: Add docstring."""  # pragma: no cover
    return load_definitions("stablehlo")

  @property  # pragma: no cover
  def specifications(self) -> Dict[str, OperationDef]:
    """TODO: Add docstring."""
    return {}  # pragma: no cover

  @property
  def rng_seed_methods(self) -> List[str]:  # pragma: no cover
    """TODO: Add docstring."""
    return []

  # pragma: no cover
  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """TODO: Add docstring."""  # pragma: no cover
    return f"// Target: {device_type}"  # pragma: no cover

  def get_device_check_syntax(self) -> str:  # pragma: no cover
    """TODO: Add docstring."""
    return "True"

  # pragma: no cover
  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """TODO: Add docstring."""
    return ""  # pragma: no cover

  def get_serialization_imports(self) -> List[str]:
    """TODO: Add docstring."""  # pragma: no cover
    return []  # pragma: no cover

  # pragma: no cover
  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:  # pragma: no cover
    """TODO: Add docstring."""
    return ""

  # pragma: no cover
  def get_weight_conversion_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """TODO: Add docstring."""
    return "# Weights not supported in StableHLO mode"  # pragma: no cover

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """TODO: Add docstring."""
    return tensor_var  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """TODO: Add docstring."""
    return "# Weights not supported in StableHLO mode"  # pragma: no cover

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """TODO: Add docstring."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """TODO: Add docstring."""
    if api_name.startswith("stablehlo."):  # pragma: no cover
      op_code = api_name.split(".")[-1]  # pragma: no cover
      return f"https://github.com/openxla/stablehlo/blob/main/docs/spec.md#{op_code}"  # pragma: no cover
    return None  # pragma: no cover

  def convert(self, data: Any) -> Any:
    """TODO: Add docstring."""
    return str(data)  # pragma: no cover

  def get_tiered_examples(self) -> Dict[str, str]:
    """TODO: Add docstring."""
    return {
      "tier1_math": "%0 = stablehlo.abs %arg0 : tensor<*xf32>",
      "tier2_neural": "module { func.func @main() { %0 = stablehlo.convolution(...) } }",
      "tier3_extras": "// Extras ignored",
    }
