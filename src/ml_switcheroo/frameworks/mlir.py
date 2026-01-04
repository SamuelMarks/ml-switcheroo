"""
MLIR (Multi-Level Intermediate Representation) Adapter.

This framework adapter treats MLIR (specifically dialects like TOSA, Linalg, or Func)
as a target "language" for textual code generation or lower-level compilation inspection.

As MLIR is an Intermediate Representation rather than a high-level execution framework
accessible comfortably via Python runtime in the same way as PyTorch/JAX, this adapter
primarily serves as a text generator or a bridge for specialized compilers.

Capabilities:
1.  **Format**: Generates MLIR textual representation strings.
2.  **Tiers**: Visual/Structural representation only.
3.  **No-Ops**: Weights handling and runtime device checks are stubbed as comments.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
  InitMode,
  OpDefinition,
)
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("mlir")
class MlirAdapter:
  """
  Adapter for generating or inspecting MLIR code.

  Usually targets dialects such as `tosa` (Tensor Operator Set Architecture)
  or `linalg` for mathematical operations.
  """

  display_name: str = "MLIR (Intermediate)"
  inherits_from: Optional[str] = None
  ui_priority: int = 90
  _mode: InitMode = InitMode.GHOST

  @property
  def search_modules(self) -> List[str]:
    """
    Returns list of search modules.
    MLIR python bindings differ constantly (e.g. `iree.compiler`, `torch_mlir`).
    Returns empty by default to treat as pure text target.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns unsafe modules.

    Returns:
        Set[str]: Empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Defines the virtual alias `sw` used in intermediate representations.

    Returns:
        Tuple[str, str]: ("mlir", "sw").
    """
    return ("mlir", "sw")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    No namespace mappings.

    Returns:
        Dict[str, ImportConfig]: Empty dict.
    """
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    No discovery heuristics.

    Returns:
        Dict[str, List[str]]: Empty dict.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Returns template for tests.
    Generates comments using MLIR syntax (`//`).

    Returns:
        Dict[str, str]: Templates.
    """
    return {
      "import": "// module attributes",
      "convert_input": "// input tensor {np_var}",
      "to_numpy": "// result tensor {res_var}",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Imports logic for harness.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Initialization logic.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Conversion to numpy.
    Returns string conversion for text compatibility.

    Returns:
        str: Python logic.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Supports Array API and Neural tiers (via Dialects).

    Returns:
        List[SemanticTier]: [ARRAY_API, NEURAL].
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    No magic args.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Structural traits (Default).

    Returns:
        StructuralTraits: Default configuration.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Plugin traits (Default).

    Returns:
        PluginTraits: Default capability flags.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for MLIR dialects.
    Loaded dynamically from `frameworks/definitions/mlir.json`.

    Returns:
        Dict[str, StandardMap]: Mappings.
    """
    return load_definitions("mlir")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """
    Returns empty specifications for MLIR adapter.

    Returns:
        Dict[str, OpDefinition]: Empty dict.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    No global seed methods.

    Returns:
        List[str]: Empty list.
    """
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    No runtime API collection.

    Args:
        category (StandardCategory): Category.

    Returns:
        List[GhostRef]: Empty list.
    """
    return []

  # --- Syntax Generation (No-Ops / Comments) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns MLIR comment regarding targeting.

    Args:
        device_type (str): Device (e.g., 'cpu').
        device_index (Optional[str]): Index.

    Returns:
        str: Comment string.
    """
    return f"// Target: {device_type}"

  def get_device_check_syntax(self) -> str:
    """
    Returns check syntax (Always True).

    Returns:
        str: "True".
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns RNG split comment.

    Args:
        rng_var (str): Source.
        key_var (str): Destination.

    Returns:
        str: Comment string.
    """
    return f"// Split RNG: {rng_var} -> {key_var}"

  def get_serialization_imports(self) -> List[str]:
    """
    Returns empty serialization imports.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns serialization syntax (No-op comments).

    Args:
        op (str): 'save' or 'load'.
        file_arg (str): Filename.
        object_arg (Optional[str]): Target object.

    Returns:
        str: Comment string.
    """
    if op == "save":
      return f"// Save {object_arg} to {file_arg}"
    return f"// Load from {file_arg}"

  # --- Weight Handling (No-Ops) ---

  def get_weight_conversion_imports(self) -> List[str]:
    """
    Imports for weight conversion script.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """
    Code to load weights (No-op).
    MLIR weights are typically `dense<>` attributes, not runtime loaded vars here.

    Args:
        path_var (str): Path variable.

    Returns:
        str: Python comment code.
    """
    return f"# Weights loading not supported in MLIR adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """
    Conversion to numpy expression.

    Args:
        tensor_var (str): Variable name.

    Returns:
        str: Identity string.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """
    Code to save weights (No-op).

    Args:
        state_var (str): State variable.
        path_var (str): Path variable.

    Returns:
        str: Python comment code.
    """
    return f"# Weights saving not supported in MLIR adapter"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual wiring.

    Args:
        snapshot (Dict[str, Any]): Snapshot dict.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns None as this dialect is undocumented.

    Args:
        api_name (str): API name.

    Returns:
        None
    """
    del api_name
    return None

  def convert(self, data: Any) -> Any:
    """
    Converts data to string representation for verification.

    Args:
        data (Any): Input.

    Returns:
        Any: String representation.
    """
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns example MLIR code.

    Returns:
        str: Code string.
    """
    return cls().get_tiered_examples()["tier1_math"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns tiered code examples.

    Returns:
        Dict[str, str]: Tier map.
    """
    return {
      "tier1_math": """// Example MLIR (Switcheroo Dialect) with AbstractOp
sw.module {
  sw.func @main(%x: !sw.type<"torch.Tensor">) {
     %0 = sw.op {type="torch.abs"} (%x)
     sw.return %0
  }
}""",
      "tier2_neural": """func.func @predict(%arg0: tensor<1x28x28x1xf32>) -> tensor<1x10xf32> {
   // Logic would involve tosa.conv2d or linalg.matmul
   return %arg0 : tensor<1x10xf32>
}""",
      "tier3_extras": "// Extras not supported",
    }
