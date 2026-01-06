"""
NVIDIA SASS (Streaming Assembler) Framework Adapter.

This module registers 'sass' as a valid source and target for the transpiler.
It integrates the SASS compiler toolchain (Parser, Synthesizer, Emitter) into
the ASTEngine via the FrameworkAdapter protocol hooks.

Capabilities:
1.  **Target (Python -> SASS)**: Converts Logical Graphs derived from Python ASTs
    into SASS Assembly text (`create_emitter`).
2.  **Source (SASS -> Python)**:
    -   **Low-Level**: Parses raw instructions into python calls (`sass.FADD(...)`)
        if no structure is detected.
    -   **High-Level (Lifting)**: Parsing semantic comment markers (e.g., `// BEGIN`)
        to reconstruct the original PyTorch model stucture (`nn.Module`).
3.  **Macro Injection**: Registers procedural macros (e.g. Conv2d loops) into
    the synthesizer to support high-level neural network operations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import libcst as cst

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
from ml_switcheroo.semantics.manager import SemanticsManager

# SASS Toolchain Imports
from ml_switcheroo.core.sass.parser import SassParser as CoreSassParser
from ml_switcheroo.core.sass.emitter import SassEmitter as CoreSassEmitter
from ml_switcheroo.core.sass.synthesizer import SassSynthesizer
from ml_switcheroo.core.sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.graph import GraphExtractor

# Lifting / Decompilation Imports
from ml_switcheroo.core.sass.lifter import SassLifter
from ml_switcheroo.core.graph_synthesizer import GraphSynthesizer
from ml_switcheroo.core.sass.nodes import Comment


class PythonToSassEmitter:
  """
  Adapter wrapper handles the conversion pipeline:
  Python Code -> Graph -> SASS AST -> SASS Text string.

  It configures the SassSynthesizer with High-Level Macros.
  """

  def __init__(self) -> None:
    """
    Initializes the pipeline components.
    Injects the Macro Registry (Conv2d, Linear) into the Synthesizer.
    """
    # We need semantics for Opcode lookup in Synthesizer
    self.semantics = SemanticsManager()
    self.synth = SassSynthesizer(self.semantics)

    # Wire Macros (1-to-N Expansion)
    # This replaces the default/empty registry logic in the synthesizer
    # ensuring that the Adapter controls the available expansions.
    self.synth.macro_registry["Conv2d"] = expand_conv2d
    self.synth.macro_registry["Linear"] = expand_linear

    self.emitter = CoreSassEmitter()

  def emit(self, code: str) -> str:
    """
    Executes emission pipeline.

    Args:
        code (str): Python source code.

    Returns:
        str: Formatted SASS assembly string.
    """
    # 1. Parse Python Code to CST
    try:
      tree = cst.parse_module(code)
    except Exception as e:
      return f"// Error parsing Python source: {e}"

    # 2. Extract Logical Graph
    extractor = GraphExtractor()
    tree.visit(extractor)
    graph = extractor.graph

    if not graph.nodes:
      return "// No logic graph extracted from source."

    # 3. Synthesize SASS AST (Using Macros for Neural Ops)
    sass_nodes = self.synth.from_graph(graph)

    # 4. Emit Text
    return self.emitter.emit(sass_nodes)


class SassToPythonParser:
  """
  Adapter wrapper handling the ingestion pipeline:
  SASS Text -> SASS AST -> Python CST.

  Implements "Lifting" logic: if semantic markers (// BEGIN ...) are detected,
  it promotes the low-level assembly back into a high-level PyTorch Module structure.
  Otherwise, it translates 1:1 to instruction calls.
  """

  def __init__(self, code: str) -> None:
    """
    Initialize the parser wrapper.

    Args:
        code (str): SASS source text.
    """
    self.code = code
    self.semantics = SemanticsManager()
    self.synth = SassSynthesizer(self.semantics)
    self.parser_core = CoreSassParser(code)

  def parse(self) -> cst.Module:
    """
    Executes parsing pipeline.

    Logic:
    1. Parse Text to SASS AST.
    2. Scan for Semantic Markers (High-Level Structure).
    3. If markers found -> Lift to Logical Graph -> Synthesize PyTorch Class.
    4. Else -> Synthesize Python AST calls for raw instructions.

    Returns:
        cst.Module: LibCST Module.
    """
    # 1. Parse SASS Text -> SASS AST
    nodes = self.parser_core.parse()

    # 2. Check for Semantic Markers (Lifting Trigger)
    has_markers = any(isinstance(n, Comment) and ("BEGIN" in n.text or "Input" in n.text) for n in nodes)

    if has_markers:
      # 3. Lifting Pipeline (Decompilation)
      lifter = SassLifter()
      graph = lifter.lift(nodes)

      # Synthesize High-Level Model Code (defaulting to PyTorch style)
      graph_synth = GraphSynthesizer(framework="torch")

      # Determine class name if possible, else default
      # Note: Lifter currently doesn't extract model names from SASS,
      # so we use a generic name.
      py_source = graph_synth.generate(graph, class_name="DecompiledModel")

      try:
        return cst.parse_module(py_source)
      except Exception:
        # Fallback if synthesis generated invalid code, though unlikely
        return self.synth.to_python(nodes)

    # 4. Low-Level Translation (Instruction Mapping)
    return self.synth.to_python(nodes)


@register_framework("sass")
class SassAdapter(FrameworkAdapter):
  """
  Adapter for NVIDIA SASS Assembly Generation.

  Provides integration points for the ASTEngine to consume or emit SASS.
  Registers procedural macros for neural layers to support 'Lowering' semantics.
  """

  display_name: str = "NVIDIA SASS"
  inherits_from: Optional[str] = None
  ui_priority: int = 150  # Low priority, low-level target
  _mode: InitMode = InitMode.GHOST

  # --- Engine Hooks ---

  def create_parser(self, code: str) -> SassToPythonParser:
    """
    Factory for the SASS Ingestion Parser.

    Args:
        code (str): The input source.

    Returns:
        SassToPythonParser: The configured parser instance.
    """
    return SassToPythonParser(code)

  def create_emitter(self) -> PythonToSassEmitter:
    """
    Factory for the SASS Target Emitter.

    Returns:
        PythonToSassEmitter: The configured emitter instance with Macros wired.
    """
    return PythonToSassEmitter()

  # --- Standard Protocol Implementation ---

  @property
  def search_modules(self) -> List[str]:
    """SASS is not a python module."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """No submodules to blacklist."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Defines the virtual alias 'asm' used in intermediate representations."""
    return ("sass", "asm")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """No namespace mappings."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """No discovery possible."""
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Returns template for tests (Comments).

    Returns:
        Dict[str, str]: Test templates.
    """
    return {
      "import": "// SASS Header",
      "convert_input": "// Input {np_var}",
      "to_numpy": "// Output {res_var}",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """No imports for execution."""
    return []

  def get_harness_init_code(self) -> str:
    """No init code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns string conversion for text compatibility logic validation."""
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Supports Array API (Math) primarily."""
    return [SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """No magic args."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """No structural traits."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """No plugin logic."""
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns mappings of Abstract Operations to SASS Implementations.
    Combines file-based definitions with hardcoded instructions/macros.

    Returns:
        Dict[str, StandardMap]: The definitions dictionary.
    """
    # Start with file-based (if any exist)
    defs = load_definitions("sass")

    # Inject Neural Macros (1-to-N) to satisfy Hub decoupling requirements
    defs["Conv2d"] = StandardMap(api="Macro.Conv2d")
    defs["Linear"] = StandardMap(api="Macro.Linear")

    # Inject Math Opcodes (1-to-1)
    defs["Add"] = StandardMap(api="FADD")
    defs["Sub"] = StandardMap(api="FADD")  # SASS often uses FADD with negation for sub
    defs["Mul"] = StandardMap(api="FMUL")
    defs["Clamp"] = StandardMap(api="MNMX")
    defs["Abs"] = StandardMap(api="IABS")

    return defs

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """No unique specifications."""
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """No RNG seeding."""
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """No API collection."""
    return []

  # --- Syntax Generation (No-Ops / Comments) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Emits comment for device selection.

    Args:
        device_type (str): CLI arg (cuda, cpu).
        device_index (Optional[str]): Index.

    Returns:
        str: Comment string.
    """
    return f"// Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    """
    Always true (virtual).

    Returns:
        str: "True".
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """No-op."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """No imports."""
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """No serialization syntax."""
    return ""

  # --- Weight Handling (No-Ops) ---

  def get_weight_conversion_imports(self) -> List[str]:
    """No imports."""
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Stub."""
    return "// Weights loading not supported in SASS adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Identity."""
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Stub."""
    return "// Weights saving not supported in SASS adapter"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No wiring logic."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """No docs URL."""
    return None

  def convert(self, data: Any) -> Any:
    """Converts data to string representation."""
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """Returns example SASS."""
    return "// Example SASS\nFADD R1, R2, R3;"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns tiered examples dict."""
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "// Neural layers map to comment blocks",
      "tier3_extras": "// Extras ignored",
    }
