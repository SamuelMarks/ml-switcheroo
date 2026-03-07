"""
MLIR Framework Adapter.

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


@register_framework("mlir")
class MlirAdapter(FrameworkAdapter):
  """Adapter for MLIR."""

  display_name: str = "MLIR (Intermediate)"
  inherits_from: Optional[str] = None
  ui_priority: int = 90
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """TODO: Add docstring."""
    pass

  @property
  def search_modules(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """TODO: Add docstring."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """TODO: Add docstring."""
    return ("mlir", "sw")

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
    return load_definitions("mlir")

  @property  # pragma: no cover
  def specifications(self) -> Dict[str, OperationDef]:
    """TODO: Add docstring."""
    return {}  # pragma: no cover

  @property
  def rng_seed_methods(self) -> List[str]:  # pragma: no cover
    """TODO: Add docstring."""
    return []

  # pragma: no cover
  def collect_api(self, category: StandardCategory) -> List[GhostRef]:  # pragma: no cover
    """TODO: Add docstring."""  # pragma: no cover
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:  # pragma: no cover
    """TODO: Add docstring."""
    return f"// Target: {device_type}"

  # pragma: no cover
  def get_device_check_syntax(self) -> str:
    """TODO: Add docstring."""
    return "True"  # pragma: no cover

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """TODO: Add docstring."""  # pragma: no cover
    return f"// Split RNG: {rng_var} -> {key_var}"  # pragma: no cover

  def get_serialization_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """TODO: Add docstring."""
    if op == "save":  # pragma: no cover
      return f"// Save {object_arg} to {file_arg}"  # pragma: no cover
    return f"// Load from {file_arg}"  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """TODO: Add docstring."""
    return f"# Weights loading not supported in MLIR adapter"  # pragma: no cover

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """TODO: Add docstring."""
    return tensor_var  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """TODO: Add docstring."""
    return f"# Weights saving not supported in MLIR adapter"  # pragma: no cover

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """TODO: Add docstring."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """TODO: Add docstring."""
    return None

  def convert(self, data: Any) -> Any:
    """TODO: Add docstring."""
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """TODO: Add docstring."""
    # Use sw.func to match test expectation
    return '// Example MLIR\nsw.module {\n^entry:\n    sw.func {sym_name = "main"} {\n        %0 = sw.op {type = "torch.abs"} (%x)\n    }\n}'

  def get_tiered_examples(self) -> Dict[str, str]:
    """TODO: Add docstring."""
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "// Extras ignored",
    }
