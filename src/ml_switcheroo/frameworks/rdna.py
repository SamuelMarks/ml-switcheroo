"""AMD RDNA / GCN Framework Adapter.

Provides metadata and configuration for the RDNA compiler stack.
This adapter acts as a metadata container for the Compiler Registry,
identifying RDNA as a target language and providing static definitions.

Migration Note:
    Legacy shim classes (`PythonToRdnaEmitter`, `RdnaToPythonParser`)
    have been removed. Routing now occurs via `compiler.registry`.
"""

from typing import Union, Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StandardMap,
  ImportConfig,
  InitMode,
  OperationDef,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
from ml_switcheroo_ir import LogicalGraph, LogicalNode

if TYPE_CHECKING:
  pass


@register_framework("rdna")
class RdnaAdapter(FrameworkAdapter):
  """Adapter for AMD RDNA.

  This adapter handles metadata registry definitions, static attributes,
  device syntax generation, code template specifications, and semantic
  decoding of raw AMD GPU assembly blocks back into logical graphs.
  """

  display_name: str = "AMD RDNA"
  inherits_from: Optional[str] = None
  ui_priority: int = 151
  _mode: InitMode = InitMode.GHOST

  def __init__(self, target_arch: str = "gfx1030") -> None:
    """Initialize the adapter.

    Args:
        target_arch: Target GPU architecture (e.g. gfx1030).
                     Note: This property is primarily informational in the new architecture.

    """
    self.target_arch = target_arch

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Get the import alias for the AMD RDNA framework.

    Returns:
        A tuple containing the import name and the alias (e.g., ("rdna", "asm")).
    """
    return "rdna", "asm"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Get the namespaces and their configurations to import for AMD RDNA.

    Returns:
        A dictionary mapping namespace names to their configurations or import configs.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Get the testing and validation configuration for AMD RDNA.

    Returns:
        A dictionary containing configuration keys like "import", "convert_input",
        and "to_numpy" mapped to their corresponding RDNA syntax/header templates.
    """
    return {"import": "; RDNA Header", "convert_input": "; Input {np_var}", "to_numpy": "; Output {res_var}"}

  @property
  def harness_imports(self) -> List[str]:
    """Get the required imports for the generated testing harness.

    Returns:
        A list of module import strings required by the RDNA test harness.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Generate initialization code for the test harness.

    Returns:
        A string of initialization instructions or setup commands.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Generate the code snippet to convert a target tensor/object to a NumPy array.

    Returns:
        A string containing the target-specific code for converting to NumPy.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the semantic tiers supported by the AMD RDNA framework.

    Returns:
        A list of SemanticTier enums representing supported abstraction tiers.
    """
    return [SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """Get the declared magic arguments supported by the adapter.

    Returns:
        A list of strings representing the magic argument names.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Get the structural traits associated with AMD RDNA.

    Returns:
        An instance of StructuralTraits defining block structure or shape patterns.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Get the plugin traits associated with AMD RDNA.

    Returns:
        An instance of PluginTraits defining code transformations or hooks.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Get the definitions and mappings for individual operations.

    Returns:
        A dictionary mapping operation names to their StandardMap definitions.
    """
    return load_definitions("rdna")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get the specifications of operations supported by the adapter.

    Returns:
        A dictionary mapping operation names to their OperationDef structures.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Get the methods or APIs used to seed the random number generator in AMD RDNA.

    Returns:
        A list of method name strings used for RNG seeding.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate the hardware syntax required to select or target a specific device.

    Args:
        device_type: The type of target device (e.g., 'gpu', 'cpu').
        device_index: Optional index of the target device.

    Returns:
        A string containing the assembly/syntax comment or instruction to target the device.
    """
    return f"; Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    """Generate the syntax/code to check if the target hardware device is available.

    Returns:
        A string representing the check statement (e.g. "True").
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generate syntax for splitting a random number generator key.

    Args:
        rng_var: The variable holding the RNG state.
        key_var: The target variable for the split key.

    Returns:
        A string representing the split operation (empty if not supported).
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Get the imports required for serialization/deserialization routines.

    Returns:
        A list of module import strings needed for serialization.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generate syntax to serialize or deserialize an object or state.

    Args:
        op: The operation name (e.g., "save", "load").
        file_arg: The filepath argument string.
        object_arg: Optional object argument to serialize.

    Returns:
        A string containing the serialization code/syntax.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Get imports required for weight format conversion.

    Returns:
        A list of import strings required for converting weight formats.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Generate code to load model weights from a given path.

    Args:
        path_var: The variable holding the filepath to load weights from.

    Returns:
        A string containing the assembly comments/code for loading weights.
    """
    return "; Weights loading not supported in RDNA adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generate an expression that converts a tensor to a NumPy array representation.

    Args:
        tensor_var: The variable name representing the tensor.

    Returns:
        A string expression that represents the NumPy conversion.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generate code to save model weights to a file.

    Args:
        state_var: The variable holding the state dictionary or weights.
        path_var: The variable holding the target file path.

    Returns:
        A string containing assembly comments/code for saving weights.
    """
    return "; Weights saving not supported in RDNA adapter"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply framework-specific wiring or configuration from a snapshot.

    Args:
        snapshot: A dictionary containing framework metadata or context configuration.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Get the documentation URL for a specific AMD RDNA API or concept.

    Args:
        api_name: The name of the API or concept.

    Returns:
        The official documentation URL string, or None if not found.
    """
    return f"https://gpuopen.com/learn/rdna-performance-guide/?q={api_name}"

  def convert(self, data: Any) -> Any:
    """Convert arbitrary data or objects into an RDNA-compatible representation.

    Args:
        data: The input data to convert.

    Returns:
        The converted RDNA-compatible data (typically a string representation).
    """
    return str(data)

  def parse_rdna_to_graph(self, rdna_code: str) -> LogicalGraph:
    """Parses `amdasm` output strings into a valid `LogicalGraph`.

    This reconstructs high-level semantics from AMD RDNA3 instruction streams.

    Args:
        rdna_code: The raw AMD assembly code string.

    Returns:
        A LogicalGraph containing reconstructed semantic nodes.
    """
    graph = LogicalGraph()
    lines = rdna_code.splitlines()

    in_loop = "s_cbranch_vccnz" in rdna_code

    for line in lines:
      line = line.strip()
      if not line or line.startswith("//"):
        continue

      if line.startswith("L_") or line.startswith("BB"):
        pass  # Label
      elif "s_cbranch_vccnz" in line:
        node = LogicalNode(id=f"node_{len(graph.nodes)}", op_type="LoopControl", attributes={"condition": line})
        graph.nodes[node.id] = node
      elif "v_fmac_f32" in line or "v_mac_f32" in line:
        if in_loop:
          node = LogicalNode(
            id=f"node_{len(graph.nodes)}",
            op_type="Conv2d",
            attributes={"inferred_from": "FMAC inside loop"},
          )
        else:
          node = LogicalNode(
            id=f"node_{len(graph.nodes)}",
            op_type="Linear",
            attributes={"inferred_from": "FMAC outside loop"},
          )
        graph.nodes[node.id] = node
    return graph

  def get_tiered_examples(self) -> Dict[str, str]:
    """Get tiered example code blocks for different semantic layers in AMD RDNA.

    Returns:
        A dictionary mapping tier names (e.g., 'tier1_math', 'tier2_neural')
        to their corresponding assembly instruction or comment examples.
    """
    return {
      "tier1_math": "; Tier 1: Core Tensor Operations\n; x = abs(x)\nv_add_f32 v0, v1, v2\n",
      "tier2_neural_simple": "; Tier 2: Neural Simple (Linear + ReLU)\n; BEGIN Linear (fc)\n; ... setup ...\nBB0_1:\n  v_mac_f32 v0, v1, v2\n  s_cbranch_vccnz BB0_1\n; END Linear\n; BEGIN ReLU\n  v_max_f32 v0, v0, 0\n; END ReLU\n",
      "tier2_neural_cnn": "; Tier 2: Neural CNN (Conv2d)\n; BEGIN Conv2d\nBB1_1:\n  v_fmac_f32 v0, v1, v2\n  s_cbranch_vccnz BB1_1\n; END Conv2d\n",
      "tier4_qwen3": "; Tier 4: Qwen3 (Linear + Multiply)\n; BEGIN Linear (down_proj)\nBB2_1:\n  v_mac_f32 v0, v1, v2\n  s_cbranch_vccnz BB2_1\n; END Linear\n; SwiGLU fusion multiply\n  v_mul_f32 v3, v4, v5\n",
      "tier4_qwen3-vl": "; Tier 4: Qwen3-VL\n; BEGIN Conv3d\nBB3_1:\n  v_fmac_f32 v0, v1, v2\n  s_cbranch_vccnz BB3_1\n; END Conv3d\n",
      "tier3_extras": "; Extras ignored",
    }
