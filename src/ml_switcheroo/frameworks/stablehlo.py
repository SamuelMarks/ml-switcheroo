"""
StableHLO Framework Adapter.

This adapter registers `stablehlo` as a valid target for the transpiler.
It treats StableHLO as a textual Intermediate Representation (IR), utilizing
the `StableHloEmitter` to generate MLIR code from Python AST.

Capabilities:
1.  **Generation**: `create_emitter` hook returns a textual emitter.
2.  **Tiers**: Supports Array and Neural tiers (via MLIR generation).
3.  **No-Ops**: Runtime hooks (weights, device checks) are stubbed or commented.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import libcst as cst

from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  ImportConfig,
  InitMode,
  StandardCategory,
  StandardMap,
  register_framework,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import OpDefinition, PluginTraits, StructuralTraits


class _StableHloTextEmitter:
  """
  Wrapper to bridge StableHloEmitter (AST->MLIR-CST) to Engine Emitter (Str->Str).
  Compatible with the `create_emitter` hook expected by `ASTEngine`.
  """

  def emit(self, code: str) -> str:
    """
    Parses Python code, resolves semantics, and emits StableHLO MLIR text.

    Args:
        code (str): The input Python source code.

    Returns:
        str: The generated MLIR string.
    """
    try:
      tree = cst.parse_module(code)
    except cst.ParserSyntaxError as e:
      return f"// Error parsing Python source: {e}"

    # Late import to prevent circular dependency lookup during framework registration
    # config -> frameworks -> stablehlo -> (core/semantics) -> config
    from ml_switcheroo.semantics.manager import SemanticsManager
    from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter

    # Note: We instantiate a fresh SemanticsManager here because the ASTEngine
    # factory interface does not currently pass the active manager to the adapter.
    # This implies a performance hit (reloading JSONs) per conversion, which is
    # accepted for the "StableHLO Extraction" use case distinct from interactive usage.
    semantics = SemanticsManager()
    emitter = StableHloEmitter(semantics)
    try:
      mlir_node = emitter.convert(tree)
      return mlir_node.to_text()
    except Exception as e:
      return f"// Error generating StableHLO: {e}"


@register_framework("stablehlo")
class StableHloAdapter:
  """
  Adapter for generating StableHLO (MLIR) text.
  """

  display_name: str = "StableHLO (MLIR)"
  inherits_from: Optional[str] = None
  ui_priority: int = 95  # Low priority (Intermediate Format)
  _mode: InitMode = InitMode.GHOST

  @property
  def search_modules(self) -> List[str]:
    """
    No modules to search; StableHLO is an IR format.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    No unsafe modules.

    Returns:
        Set[str]: Empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Virtual import alias matching standard textual representation.

    Returns:
        Tuple[str, str]: ("stablehlo", "stablehlo").
    """
    return ("stablehlo", "stablehlo")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    No namespaces to verify for import injection.

    Returns:
        Dict[str, ImportConfig]: Empty dict.
    """
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    No discovery possible.

    Returns:
        Dict[str, List[str]]: Empty dict.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Placeholder configuration for test generation.
    Generates MLIR comments instead of Python logic.

    Returns:
        Dict[str, str]: Comment templates.
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
    Returns empty imports.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Returns empty init code.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns generic stringifier for verification compat.

    Returns:
        str: Code returning string representation.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Supports Neural and Array tiers for IR generation.

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
    Defines default structural traits.

    Returns:
        StructuralTraits: Default configuration.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Defines default capabilities.

    Returns:
        PluginTraits: Default traits.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns static definitions for StableHLO mappings.
    Loaded dynamically from `frameworks/definitions/stablehlo.json`.

    Returns:
        Dict[str, StandardMap]: Mappings for translation logic.
    """
    return load_definitions("stablehlo")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """
    Returns empty specifications.

    Returns:
        Dict[str, OpDefinition]: Empty dict.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    No RNG logic.

    Returns:
        List[str]: Empty list.
    """
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    No API to collect.

    Args:
        category (StandardCategory): Ignored.

    Returns:
        List[GhostRef]: Empty list.
    """
    return []

  # --- Syntax Generation (No-Ops) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns device comment.

    Args:
        device_type (str): Device.
        device_index (Optional[str]): Index.

    Returns:
        str: MLIR comment.
    """
    return f"// Target: {device_type}"

  def get_device_check_syntax(self) -> str:
    """
    Returns generic boolean.

    Returns:
        str: "True".
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns RNG comment.

    Args:
        rng_var (str): Source.
        key_var (str): Destination.

    Returns:
        str: Empty string or comment.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """
    Empty imports.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns empty string for IO.

    Args:
        op (str): Operation.
        file_arg (str): Filename.
        object_arg (Optional[str]): Object.

    Returns:
        str: Empty string.
    """
    return ""

  # --- Weight Handling (No-Ops) ---

  def get_weight_conversion_imports(self) -> List[str]:
    """
    Empty list.

    Returns:
        List[str]: Empty.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """
    Returns comment code.

    Args:
        path_var (str): Path.

    Returns:
        str: Comment.
    """
    return f"# Weights not supported in StableHLO mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """
    Identity.

    Args:
        tensor_var (str): Var name.

    Returns:
        str: Var name.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """
    Returns comment code.

    Args:
        state_var (str): State.
        path_var (str): Path.

    Returns:
        str: Comment.
    """
    return f"# Weights not supported in StableHLO mode"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    No wiring needed.

    Args:
        snapshot (Dict[str, Any]): Snapshot dict.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns link to OpenXLA spec if applicable.

    Args:
        api_name (str): API name (e.g. stablehlo.abs).

    Returns:
        Optional[str]: URL.
    """
    if api_name.startswith("stablehlo."):
      op_code = api_name.split(".")[-1]
      return f"https://github.com/openxla/stablehlo/blob/main/docs/spec.md#{op_code}"
    return None

  def convert(self, data: Any) -> Any:
    """
    Returns string representation of data.

    Args:
        data (Any): Input.

    Returns:
        Any: Stringified input.
    """
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns example MLIR snippet.

    Returns:
        str: MLIR code.
    """
    return "%0 = stablehlo.abs %arg0 : tensor<*xf32>"

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns example strings.

    Returns:
        Dict[str, str]: Examples.
    """
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "module { func.func @main() { %0 = stablehlo.convolution(...) } }",
      "tier3_extras": "// Extras ignored",
    }

  # --- Engine Factories ---

  def create_emitter(self) -> Any:
    """
    Factory for the output generator.

    Returns:
        _StableHloTextEmitter: Wrapper to emit string source code.
    """
    return _StableHloTextEmitter()
