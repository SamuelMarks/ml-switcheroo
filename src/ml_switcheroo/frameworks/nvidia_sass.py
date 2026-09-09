"""NVIDIA SASS (Streaming Assembler) Framework Adapter.

Provides metadata and configuration for the NVIDIA_SASS compiler stack.
This adapter acts as a metadata container for the Compiler Registry,
identifying NVIDIA_SASS as a target language and providing static definitions.

Migration Note:
    Legacy shim classes (`PythonToNvidiaSassEmitter`) have been removed.
    Compilation logic is now handled by `ml_switcheroo.core.compiler.backends.nvidia_sass.NvidiaSassBackend`.
"""

import typing


from typing import Union, Dict, List, Optional, Tuple, TYPE_CHECKING
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


@register_framework("nvidia_sass")
class NvidiaSassAdapter(FrameworkAdapter):
  """Adapter for NVIDIA SASS Assembly Generation."""

  display_name: str = "NVIDIA SASS"
  inherits_from: Optional[str] = None
  ui_priority: int = 150
  _mode: InitMode = InitMode.GHOST

  def __init__(self, target_arch: str = "sm_80") -> None:
    """Initialize the adapter.

    Args:
        target_arch: Target GPU architecture (e.g. sm_80).
                     Note: This property is primarily informational in the new architecture.
    """
    self.target_arch = target_arch

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Get the preferred import name and alias for NVIDIA_SASS in generated files.

    Returns:
        A tuple of (import_name, alias) indicating how NVIDIA_SASS is referenced.
    """
    return "nvidia_sass", "asm"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Get the namespace structure and import configurations for NVIDIA_SASS.

    Returns:
        A dictionary mapping namespace paths to their import configuration schemas.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Get the configuration dictionary required for NVIDIA_SASS code generation tests.

    Returns:
        A dictionary with NVIDIA_SASS code templates such as "import", "convert_input",
        and "to_numpy" expressions.
    """
    return {"import": "// NVIDIA_SASS Header", "convert_input": "// Input {np_var}", "to_numpy": "// Output {res_var}"}

  @property
  def harness_imports(self) -> List[str]:
    """Get the list of imports needed by the testing/harness environment.

    Returns:
        A list of python import statement strings required for NVIDIA_SASS harnesses.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Get the initialization code block required for the test harness.

    Returns:
        A string of setup/initialization code for execution environments.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Get python code to convert a NVIDIA_SASS-related object to a NumPy array.

    Returns:
        A string containing Python code that converts the output object into standard representation.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the semantic tiers supported by the NVIDIA_SASS framework adapter.

    Returns:
        A list of supported SemanticTier enum values.
    """
    return [SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """Get magic arguments declared/registered specifically for NVIDIA_SASS generation.

    Returns:
        A list of supported magic argument name strings.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Get structural traits defining capability limitations or capabilities of NVIDIA_SASS.

    Returns:
        A StructuralTraits object containing structural metadata.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Get plugins/hooks capabilities supported by this adapter.

    Returns:
        A PluginTraits object detailing plugin configurations.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Load and return standard mapping definitions associated with NVIDIA_SASS.

    Returns:
        A dictionary mapping operation identifiers to StandardMap definitions.
    """
    return load_definitions("nvidia_sass")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get specific operation schemas or specifications for NVIDIA_SASS.

    Returns:
        A dictionary of operation IDs mapped to their OperationDef objects.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Get methods supported by NVIDIA_SASS for setting random number generator seeds.

    Returns:
        A list of string names for RNG seeding methods.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Get the NVIDIA_SASS syntax representing device context setup or selection.

    Args:
        device_type: The target device category (e.g., 'cuda', 'cpu').
        device_index: Optional device index/identifier.

    Returns:
        A string containing device syntax or device comment representation.
    """
    return f"// Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    """Get python expression string to verify if targeted device environment is available.

    Returns:
        A python expression string returning a boolean value.
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Get split logic syntax for random number generation keys in NVIDIA_SASS.

    Args:
        rng_var: Variable name of the source RNG state.
        key_var: Variable name to assign the resulting key state.

    Returns:
        A string representation of the key-splitting code block.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Get serialization package imports required for serialization operations.

    Returns:
        A list of import statement strings.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Get syntax representing serialization or deserialization of models.

    Args:
        op: The serialization operation type (e.g., 'load', 'save').
        file_arg: Target file path argument.
        object_arg: Optional model or state object to serialize.

    Returns:
        A string containing serialization command code.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Get module imports required for weight/parameter conversions.

    Returns:
        A list of import statements.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Get code representation for loading model weights from a specified path.

    Args:
        path_var: Python variable name containing the model checkpoint path.

    Returns:
        A string code block performing weights loading.
    """
    return "// Weights loading not supported in NVIDIA_SASS adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Get python expression string converting a NVIDIA_SASS tensor structure into NumPy.

    Args:
        tensor_var: Variable name representing the tensor to convert.

    Returns:
        A string representing the conversion expression.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Get code representation for saving model weights/state to a specified path.

    Args:
        state_var: Variable name containing the state to save.
        path_var: Variable name containing the destination file path.

    Returns:
        A string code block performing the state save operation.
    """
    return "// Weights saving not supported in NVIDIA_SASS adapter"

  def apply_wiring(self, snapshot: typing.Dict[str, dict]) -> None:
    """Apply framework-specific wiring or context snapshot logic.

    Args:
        snapshot: A dictionary representing the framework state or snapshot.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Get official documentation URL for a given API method or instruction.

    Args:
        api_name: Name of the instruction or API endpoint.

    Returns:
        The URL string if found, otherwise None.
    """
    return None

  def convert(
    self, data: typing.Union[int, float, str, list, dict]
  ) -> typing.Union[int, float, str, list, dict, typing.Any]:
    """Convert input data into a format suitable for the NVIDIA_SASS adapter.

    Args:
        data: Input data object of any type.

    Returns:
        The converted representation (typically a string).
    """
    return str(data)

  def parse_nvidia_sass_to_graph(self, sass_code: str) -> LogicalGraph:
    """Parse `cuobjdump` NVIDIA_SASS output strings into a valid `LogicalGraph`.

    This uses proper CFG reconstruction and dominator analysis rather than
    naive string heuristics.

    Args:
        sass_code: Raw NVIDIA_SASS assembly text from cuobjdump.

    Returns:
        A reconstructed LogicalGraph representing high-level semantics.
    """
    from ml_switcheroo.analysis.cfg import ControlFlowGraph, BasicBlock
    from ml_switcheroo.analysis.dominators import build_dominator_sets, find_back_edges

    cfg = ControlFlowGraph()
    current_block: Optional[BasicBlock] = None

    # Simple linear parser to build blocks
    lines = sass_code.splitlines()
    for line in lines:
      line = line.strip()
      if not line or line.startswith("//"):
        continue

      if line.startswith("L_") or line.startswith("BB"):
        # Label means new block
        label_name = line.strip(":")
        new_block = cfg.get_or_create_block(label_name)
        if current_block:
          # Fallthrough
          current_block.add_successor(new_block)
        current_block = new_block
      else:
        if current_block is None:
          current_block = cfg.get_or_create_block("entry")

        current_block.add_instruction(line)

        # Handle explicit branches
        if "BRA " in line:
          # Extract target label
          parts = line.split("BRA ")
          if len(parts) > 1:  # pragma: no branch
            target = parts[1].strip().strip(";")
            target_block = cfg.get_or_create_block(target)
            current_block.add_successor(target_block)

    # If no blocks were created (empty), return empty graph
    if not cfg.blocks:
      return LogicalGraph()

    # Analyze for loops using dominators
    doms = build_dominator_sets(cfg)
    back_edges = find_back_edges(cfg, doms)

    # Determine nodes in loops
    loop_nodes = set()
    for src, dst in back_edges:
      loop_nodes.add(src)
      loop_nodes.add(dst)

    # Map CFG back to LogicalGraph for return
    graph = LogicalGraph()
    for block_id, block in cfg.blocks.items():
      # Heuristic pattern matching on the block level
      is_in_loop = block_id in loop_nodes
      has_ffma = any("FFMA" in inst for inst in block.instructions)

      if is_in_loop and has_ffma:
        node = LogicalNode(
          id=f"node_{len(graph.nodes)}", op_type="Conv2d", attributes={"inferred_from": "FFMA inside loop"}
        )
        graph.nodes[node.id] = node
      elif has_ffma:
        node = LogicalNode(
          id=f"node_{len(graph.nodes)}", op_type="Linear", attributes={"inferred_from": "FFMA outside loop"}
        )
        graph.nodes[node.id] = node

      # Just tracking loops as control flow points if no ops match
      if is_in_loop and not has_ffma:
        node = LogicalNode(id=f"node_{len(graph.nodes)}", op_type="LoopControl", attributes={"block": block_id})
        graph.nodes[node.id] = node

    return graph

  def get_tiered_examples(self) -> Dict[str, str]:
    """Provide representative NVIDIA_SASS assembly code for different tiers of operations.

    Returns:
        A dictionary mapping tier names to code snippets.
    """
    return {
      "tier1_math": """// Tier 1: Core Tensor Operations
// x = abs(x)
FABS R1, R1;
// b = add(a, y)
FADD R2, R1, R3;

// mean(b)
// BEGIN Mean (mean)
MOV R4, RZ;
MOV R5, RZ;
L_MEAN_mean:
LDG.E.F32 R6, [R2];
FADD R4, R4, R6;
IADD3 R2, R2, 4, RZ;
IADD3 R5, R5, 1, RZ;
ISETP.LT.AND P0, PT, R5, 128, PT;
@P0 BRA L_MEAN_mean;
MOV R7, 0.0078125;
FMUL R4, R4, R7;
// END Mean (mean)
""",
      "tier2_neural_simple": """// Tier 2: Neural Simple (Linear + ReLU)
// BEGIN Linear (fc)
MOV R1, RZ;
MOV R2, RZ;
L_GEMM_fc:
LDG.E.F32 R3, [R4];
LDG.E.F32 R5, [R6];
FFMA R1, R3, R5, R1;
IADD3 R4, R4, 4, RZ;
IADD3 R6, R6, 4, RZ;
IADD3 R2, R2, 1, RZ;
ISETP.LT.AND P0, PT, R2, 10, PT;
@P0 BRA L_GEMM_fc;
// END Linear (fc)

// BEGIN ReLU (relu)
FMAX R1, R1, RZ;
// END ReLU (relu)
""",
      "tier2_neural_cnn": """// Tier 2: Neural CNN (Conv2d + Flatten + Linear)
// BEGIN Conv2d (conv)
MOV R1, RZ;
MOV R2, RZ;
L_KY_conv:
MOV R3, RZ;
L_KX_conv:
// Calc Address & Load Image Pixel
IMAD R4, R2, 32, R5;
IADD3 R4, R4, R3, RZ;
LDG.E.F32 R6, [R4];
// Calc Address & Load Weight
IMAD R4, R2, 16, R7;
IADD3 R4, R4, R3, RZ;
LDG.E.F32 R8, [R4];
FFMA R1, R6, R8, R1;
IADD3 R3, R3, 1, RZ;
ISETP.LT.AND P0, PT, R3, 3, PT;
@P0 BRA L_KX_conv;
IADD3 R2, R2, 1, RZ;
ISETP.LT.AND P0, PT, R2, 3, PT;
@P0 BRA L_KY_conv;
// END Conv2d (conv)

// BEGIN Flatten (flatten)
MOV R10, R1;
// END Flatten (flatten)
""",
      "tier4_qwen3": """// Tier 4: Qwen3 (Linear + Multiply)
// BEGIN Linear (down_proj)
MOV R1, RZ;
MOV R2, RZ;
L_GEMM_down_proj:
LDG.E.F32 R3, [R4];
LDG.E.F32 R5, [R6];
FFMA R1, R3, R5, R1;
IADD3 R4, R4, 4, RZ;
IADD3 R6, R6, 4, RZ;
IADD3 R2, R2, 1, RZ;
ISETP.LT.AND P0, PT, R2, 4096, PT;
@P0 BRA L_GEMM_down_proj;
// END Linear (down_proj)

// SwiGLU fusion multiply
FMUL R7, R8, R9;
""",
      "tier4_qwen3-vl": """// Tier 4: Qwen3-VL (Reshape + Conv3d)
// BEGIN Reshape (reshape)
MOV R2, R1;
// END Reshape (reshape)

// BEGIN Conv3d (proj)
MOV R3, RZ;
MOV R4, RZ;
L_KZ_proj:
MOV R5, RZ;
L_KY_proj:
MOV R6, RZ;
L_KX_proj:
IMAD R7, R4, 64, R8;
IMAD R7, R5, 32, R7;
IADD3 R7, R7, R6, RZ;
LDG.E.F32 R9, [R7];
IMAD R7, R4, 32, R10;
IMAD R7, R5, 16, R7;
IADD3 R7, R7, R6, RZ;
LDG.E.F32 R11, [R7];
FFMA R3, R9, R11, R3;
IADD3 R6, R6, 1, RZ;
ISETP.LT.AND P0, PT, R6, 14, PT;
@P0 BRA L_KX_proj;
IADD3 R5, R5, 1, RZ;
ISETP.LT.AND P0, PT, R5, 14, PT;
@P0 BRA L_KY_proj;
IADD3 R4, R4, 1, RZ;
ISETP.LT.AND P0, PT, R4, 2, PT;
@P0 BRA L_KZ_proj;
// END Conv3d (proj)
""",
    }
