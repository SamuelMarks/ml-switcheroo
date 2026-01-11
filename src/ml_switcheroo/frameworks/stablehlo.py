"""
StableHLO Framework Adapter.

Provides metadata and hooks for the MLIR/StableHLO stack.
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
from ml_switcheroo.semantics.schema import OpDefinition, PluginTraits, StructuralTraits
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter


class PythonToStableHloEmitter:
  """
  Wrapper to adapt the CST-based StableHloEmitter to the simpler emit() protocol
  expected by tests/engine logic if used via fallback.
  """

  def __init__(self):
    # Lazy import prevents circular dependency when initializing framework registry
    from ml_switcheroo.semantics.manager import SemanticsManager

    self.semantics = SemanticsManager()
    self.emitter = StableHloEmitter(self.semantics)

  def emit(self, code: str) -> str:
    import libcst as cst

    try:
      tree = cst.parse_module(code)
      mlir_node = self.emitter.convert(tree)
      return mlir_node.to_text()
    except Exception as e:
      return f"// Error parsing Python source: {e}"


@register_framework("stablehlo")
class StableHloAdapter(FrameworkAdapter):
  """Adapter for StableHLO."""

  display_name: str = "StableHLO (MLIR)"
  inherits_from: Optional[str] = None
  ui_priority: int = 95
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    pass

  def create_emitter(self) -> PythonToStableHloEmitter:
    """Factory for the StableHLO Emitter."""
    return PythonToStableHloEmitter()

  @property
  def search_modules(self) -> List[str]:
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("stablehlo", "stablehlo")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    return {
      "import": "// module attributes",
      "convert_input": "// input tensor {np_var}",
      "to_numpy": "// result tensor {res_var}",
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
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]

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
    return load_definitions("stablehlo")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return f"// Target: {device_type}"

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
    return "# Weights not supported in StableHLO mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    return "# Weights not supported in StableHLO mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    if api_name.startswith("stablehlo."):
      op_code = api_name.split(".")[-1]
      return f"https://github.com/openxla/stablehlo/blob/main/docs/spec.md#{op_code}"
    return None

  def convert(self, data: Any) -> Any:
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    return "%0 = stablehlo.abs %arg0 : tensor<*xf32>"

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "module { func.func @main() { %0 = stablehlo.convolution(...) } }",
      "tier3_extras": "// Extras ignored",
    }
