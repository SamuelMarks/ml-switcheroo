"""
AMD RDNA / GCN Framework Adapter.

This module registers 'rdna' as a valid source and target for the transpiler.
It integrates the RDNA compiler toolchain (Nodes, Parser, Emitter, Synthesizer,
Lifter, Macros) into the ASTEngine via the FrameworkAdapter protocol hooks.

This provides a complete compiler pipeline:
1.  **Compiler (Python -> RDNA)**: Lowers PyTorch `nn.Module` classes into
    RDNA assembly code with macro-expanded loops for layers.
2.  **Decompiler (RDNA -> Python)**: Lifts RDNA assembly back into high-level
    Python `nn.Module` code using semantic markers and instruction analysis.

Capabilities:
1.  **Lowering**: Graph Extraction -> Macro Expansion -> Register Allocation -> Assembly.
2.  **Lifting**: Assembly Parsing -> Block Analysis -> Graph Reconstruction -> Code Synthesis.
3.  **Instruction Mapping**: 1:1 mapping for arithmetic ops via Semantic Knowledge Base.
"""

import re
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

# Toolchain Imports
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.rdna.parser import RdnaParser
from ml_switcheroo.core.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.rdna.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.rdna.nodes import Comment

# Lifting / Decompilation Imports
from ml_switcheroo.core.rdna.lifter import RdnaLifter
from ml_switcheroo.core.graph_synthesizer import GraphSynthesizer


class PythonToRdnaEmitter:
  """
  Adapter wrapper handling the conversion pipeline:
  Python Code -> Graph -> RDNA AST -> RDNA Assembly Text.

  Utilizes `GraphExtractor` to build a LogicalGraph from Python source,
  and `RdnaSynthesizer` to lower that graph into assembly instructions,
  injecting Neural Network macros (Conv2d, Linear) where applicable.
  """

  def __init__(self, target_arch: str = "gfx1030") -> None:
    """
    Initializes the pipeline components.

    Args:
        target_arch: The target GPU microarchitecture (e.g. 'gfx1030').
                     Used to toggle instruction support (e.g. v_fmac vs v_fma).
    """
    self.target_arch = target_arch

    # Initialize Semantics and Synthesizer
    self.semantics = SemanticsManager()
    self.synth = RdnaSynthesizer(self.semantics)

    # Wire Macros (Architectural Lowering logic)
    self.synth.macro_registry["Conv2d"] = expand_conv2d
    self.synth.macro_registry["Linear"] = expand_linear

    self.emitter = RdnaEmitter()

  def emit(self, code: str) -> str:
    """
    Executes the compilation pipeline.

    1.  **Parse**: Convert Python string to CST.
    2.  **Extract**: Build a LogicalGraph of layers and data flow.
    3.  **Synthesize**: Convert Graph nodes to RDNA Instructions (with Register Allocation).
    4.  **Emit**: Format instructions into Assembly text.

    Args:
        code: Python source code string.

    Returns:
        str: Formatted RDNA assembly string with semantic comments.
    """
    # 1. Parse Python to CST
    try:
      tree = cst.parse_module(code)
    except Exception as e:
      return f"; Error parsing Python source: {e}\n"

    # 2. Extract Logical Graph
    extractor = GraphExtractor()
    tree.visit(extractor)
    graph = extractor.graph

    if not graph.nodes:
      return "; No logic graph extracted from source.\n"

    # 3. Synthesize RDNA AST
    # (This handles Register Allocation and Macro Expansion)
    rdna_nodes = self.synth.from_graph(graph)

    # 4. Emit Text
    header = f"; RDNA Code Generation Initialized (Arch: {self.target_arch})\n"
    assembly = self.emitter.emit(rdna_nodes)
    return header + assembly


class RdnaToPythonParser:
  """
  Adapter wrapper handling the ingestion pipeline:
  RDNA Text -> RDNA AST -> Python CST.

  Implements a "Lifting" pipeline to reconstruct high-level Python code
  from assembly:
  1.  **Sanitization**: Fixes lexer incompatibilities (e.g., `vmcnt(0)`).
  2.  **Parsing**: Converts text to RDNA AST.
  3.  **Lifting**: Detects Semantic Markers to reconstruct the Logical Graph.
  4.  **Synthesis**: Generates a Python `nn.Module` class from the Graph.
  """

  def __init__(self, code: str) -> None:
    """
    Initialize the parser wrapper.

    Args:
        code: RDNA source text.
    """
    # Workaround for Lexer limitations regarding parens in vmcnt(0)
    # We normalize `vmcnt(0)` -> `vmcnt_0` to parse as a valid identifier
    self.code = re.sub(r"vmcnt\((\d+)\)", r"vmcnt_\1", code)
    self.parser_core = RdnaParser(self.code)

  def parse(self) -> cst.Module:
    """
    Executes parsing and lifting pipeline.

    Logic:
    1.  Parse RDNA Text to RDNA AST Nodes.
    2.  Scan for Semantic Markers (`BEGIN`, `Input`) added by the Compiler.
    3.  If found -> **Decompile**:
        a. Run `RdnaLifter` to recover Layer topology and Parameters.
        b. Run `GraphSynthesizer` to generate PyTorch class structure.
    4.  Else -> **Translation**:
        a. Run `RdnaSynthesizer.to_python` to generate low-level checks.

    Returns:
        cst.Module: LibCST Module representing the reconstructed logic.
    """
    # 1. Parse
    try:
      nodes = self.parser_core.parse()
    except Exception:
      # Fallback for empty or invalid input
      return cst.Module(body=[])

    # 2. Check for High-Level Markers
    has_markers = any(isinstance(n, Comment) and ("BEGIN" in n.text or "Input" in n.text) for n in nodes)

    if has_markers:
      # 3. Lift / Decompile
      lifter = RdnaLifter()
      try:
        graph = lifter.lift(nodes)
        # Synthesize Python Class (Defaulting to Torch syntax)
        # Reconstructs: class DecompiledNet(nn.Module)
        graph_synth = GraphSynthesizer(framework="torch")
        py_source = graph_synth.generate(graph, class_name="DecompiledNet")
        return cst.parse_module(py_source)
      except Exception:
        # If lifting logic fails (broken graph), fall back to low-level translation
        pass

    # 4. Low-Level Translation (Instruction Mapping)
    semantics = SemanticsManager()
    synth = RdnaSynthesizer(semantics)
    return synth.to_python(nodes)


@register_framework("rdna")
class RdnaAdapter(FrameworkAdapter):
  """
  Adapter for AMD RDNA / GCN Assembly Generation.

  Branding:
  - Source: RDNA Assembly (Text)
  - Target: RDNA Assembly (Text)

  Attributes:
      target_arch: The default target microarchitecture (e.g., "gfx1030").
  """

  display_name: str = "AMD RDNA"
  inherits_from: Optional[str] = None
  ui_priority: int = 151  # Low priority (hardware target)
  _mode: InitMode = InitMode.GHOST

  def __init__(self, target_arch: str = "gfx1030") -> None:
    """
    Initialize the adapter.
    Updated to default to `gfx1030` matching test expectations.

    Args:
        target_arch: Target GPU generation string.
    """
    self.target_arch = target_arch

  # --- Engine Hooks ---

  def create_parser(self, code: str) -> RdnaToPythonParser:
    """
    Factory for the RDNA Ingestion Parser (Source).

    Args:
        code: The input source.

    Returns:
        RdnaToPythonParser: The configured parser instance.
    """
    return RdnaToPythonParser(code)

  def create_emitter(self) -> PythonToRdnaEmitter:
    """
    Factory for the RDNA Target Emitter (Target).

    Returns:
        PythonToRdnaEmitter: The configured emitter instance.
    """
    return PythonToRdnaEmitter(target_arch=self.target_arch)

  # --- Standard Protocol Implementation ---

  @property
  def search_modules(self) -> List[str]:
    """Returns module search paths (Empty for ISA)."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Returns unsafe modules to skip (Empty)."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Defines the virtual alias 'rdna' used in intermediate representations."""
    return ("rdna", "asm")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Returns namespace mappings (Empty)."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Patterns for auto-discovery (Empty)."""
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Returns template for generated tests."""
    return {
      "import": "; RDNA Header",
      "convert_input": "; Input {np_var}",
      "to_numpy": "; Output {res_var}",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """No imports for execution."""
    return []

  def get_harness_init_code(self) -> str:
    """No initialization code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns code to normalize output (Stringification)."""
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Returns supported semantic tiers ([ARRAY_API])."""
    return [SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """No magic args."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Structural rewriting configuration."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Plugin behavior configuration."""
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns mappings of Abstract Operations to RDNA Implementations.
    Loaded dynamically from `frameworks/definitions/rdna.json`.
    """
    return load_definitions("rdna")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """Returns unique Abstract Operations defined by this framework."""
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """No RNG seeding."""
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """No API collection possible for an ISA."""
    return []

  # --- Syntax Generation (No-Ops / Comments) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Emits comment for device selection."""
    return f"; Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    """Always true (virtual)."""
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
    return "; Weights loading not supported in RDNA adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Identity."""
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Stub."""
    return "; Weights saving not supported in RDNA adapter"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies manual wiring logic."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Returns documentation URL for an opcode."""
    return f"https://gpuopen.com/learn/rdna-performance-guide/?q={api_name}"

  def convert(self, data: Any) -> Any:
    """Converts data to string representation."""
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """Returns example RDNA."""
    return "v_add_f32 v0, v1, v2"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns tiered examples dict."""
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "; Neural layers map to comment blocks",
      "tier3_extras": "; Extras ignored",
    }
