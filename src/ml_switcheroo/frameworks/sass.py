"""
NVIDIA SASS (Streaming Assembler) Framework Adapter.

This module registers 'sass' as a valid source and target for the transpiler.
It integrates the SASS compiler toolchain (Parser, Synthesizer, Emitter) into
the ASTEngine via the FrameworkAdapter protocol hooks.

Capabilities:
1.  **Target (Python -> SASS)**: Converts Logical Graphs derived from Python ASTs
    into SASS Assembly text (`create_emitter`).
2.  **Source (SASS -> Python)**: Parses SASS Assembly text into Python ASTs
    representing instruction calls (`create_parser`).
3.  **Definitions**: Maps abstract mathematical operations (e.g. ``Add``) to
    SASS mnemonics (e.g. ``FADD``) via ``frameworks/definitions/sass.json``.
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
from ml_switcheroo.core.graph import GraphExtractor


class PythonToSassEmitter:
  """
  Adapter wrapper handles the conversion pipeline:
  Python Code -> Graph -> SASS AST -> SASS Text string.
  """

  def __init__(self):
    # We need semantics for Opcode lookup in Synthesizer
    self.semantics = SemanticsManager()
    self.synth = SassSynthesizer(self.semantics)
    self.emitter = CoreSassEmitter()

  def emit(self, code: str) -> str:
    """
    Executes emission pipeline.

    Args:
        code: Python source code.

    Returns:
        Formatted SASS assembly string.
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

    # 3. Synthesize SASS AST
    sass_nodes = self.synth.from_graph(graph)

    # 4. Emit Text
    return self.emitter.emit(sass_nodes)


class SassToPythonParser:
  """
  Adapter wrapper handling the ingestion pipeline:
  SASS Text -> SASS AST -> Python CST.
  """

  def __init__(self, code: str):
    self.code = code
    self.semantics = SemanticsManager()
    self.synth = SassSynthesizer(self.semantics)
    self.parser_core = CoreSassParser(code)

  def parse(self) -> cst.Module:
    """
    Executes parsing pipeline.

    Returns:
        LibCST Module containing synthesized Python code (e.g. `R0 = sass.FADD(...)`).
    """
    # 1. Parse SASS Text -> SASS AST
    nodes = self.parser_core.parse()

    # 2. Synthesize Python AST -> LibCST Module
    return self.synth.to_python(nodes)


@register_framework("sass")
class SassAdapter(FrameworkAdapter):
  """
  Adapter for NVIDIA SASS Assembly Generation.

  Provides integration points for the ASTEngine to consume or emit SASS.
  """

  display_name: str = "NVIDIA SASS"
  inherits_from: Optional[str] = None
  ui_priority: int = 150  # Low priority, low-level target
  _mode: InitMode = InitMode.GHOST

  # --- Engine Hooks ---

  def create_parser(self, code: str) -> SassToPythonParser:
    """Factory for the SASS Ingestion Parser."""
    return SassToPythonParser(code)

  def create_emitter(self) -> PythonToSassEmitter:
    """Factory for the SASS Target Emitter."""
    return PythonToSassEmitter()

  # --- Standard Protocol Implementation ---

  @property
  def search_modules(self) -> List[str]:
    """SASS is not a python module."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
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
    """Returns template for tests (Comments)."""
    return {
      "import": "// SASS Header",
      "convert_input": "// Input {np_var}",
      "to_numpy": "// Output {res_var}",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    return []

  def get_harness_init_code(self) -> str:
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
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for SASS mnemonics.
    Loaded dynamically from ``frameworks/definitions/sass.json``.
    """
    return load_definitions("sass")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  # --- Syntax Generation (No-Ops / Comments) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return f"// Target Device: {device_type}"

  def get_device_check_syntax(self) -> str:
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    return ""

  def get_serialization_imports(self) -> List[str]:
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    return ""

  # --- Weight Handling (No-Ops) ---

  def get_weight_conversion_imports(self) -> List[str]:
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    return "// Weights loading not supported in SASS adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    return "// Weights saving not supported in SASS adapter"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    return None

  def convert(self, data: Any) -> Any:
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    return "// Example SASS\nFADD R1, R2, R3;"

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "// Neural layers map to comment blocks",
      "tier3_extras": "// Extras ignored",
    }
