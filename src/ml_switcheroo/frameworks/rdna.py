"""
AMD RDNA / GCN Framework Adapter.

Provides metadata and legacy hooks for the RDNA compiler stack.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

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

# Import compiler components for legacy wrappers
from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.compiler.frontends.python import PythonFrontend

if TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class PythonToRdnaEmitter:
  """
  Legacy wrapper for compilation pipeline (Python -> RDNA).
  Matches interface expected by old tests.
  """

  def __init__(self, target_arch: str = "gfx1030"):
    # Lazy Import to break cycle
    from ml_switcheroo.semantics.manager import SemanticsManager

    self.target_arch = target_arch
    self.semantics = SemanticsManager()
    self.synthesizer = RdnaSynthesizer(self.semantics)
    self.emitter = RdnaEmitter()

  def emit(self, code: str) -> str:
    # 1. Parse Python to Graph
    frontend = PythonFrontend(code)
    graph = frontend.parse_to_graph()

    # 2. Synthesize RDNA AST
    rdna_nodes = self.synthesizer.from_graph(graph)

    # 3. Emit Text
    body = self.emitter.emit(rdna_nodes)
    header = f"; RDNA Code Generation Initialized (Arch: {self.target_arch})\n"
    return header + body


class RdnaToPythonParser:
  """
  Legacy wrapper for decompilation pipeline (RDNA -> Python).
  """

  def __init__(self, code: str):
    self.code = code
    self.parser = RdnaParser(code)
    self.lifter = RdnaLifter()

  def parse(self) -> cst.Module:
    # 1. Parse RDNA to AST
    from ml_switcheroo.semantics.manager import SemanticsManager

    nodes = self.parser.parse()

    # 2. Lift to Graph
    synth = RdnaSynthesizer(SemanticsManager())
    return synth.to_python(nodes)


@register_framework("rdna")
class RdnaAdapter(FrameworkAdapter):
  """Adapter for AMD RDNA."""

  display_name: str = "AMD RDNA"
  inherits_from: Optional[str] = None
  ui_priority: int = 151
  _mode: InitMode = InitMode.GHOST

  def __init__(self, target_arch: str = "gfx1030") -> None:
    self.target_arch = target_arch

  def create_emitter(self) -> PythonToRdnaEmitter:
    """Factory for legacy emitter shim."""
    return PythonToRdnaEmitter(target_arch=self.target_arch)

  def create_parser(self, code: str) -> RdnaToPythonParser:
    """Factory for legacy parser shim."""
    return RdnaToPythonParser(code)

  @property
  def search_modules(self) -> List[str]:
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("rdna", "asm")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    return {
      "import": "; RDNA Header",
      "convert_input": "; Input {np_var}",
      "to_numpy": "; Output {res_var}",
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
    return load_definitions("rdna")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return f"; Target Device: {device_type}"

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
    return "; Weights loading not supported in RDNA adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    return "; Weights saving not supported in RDNA adapter"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    return f"https://gpuopen.com/learn/rdna-performance-guide/?q={api_name}"

  def convert(self, data: Any) -> Any:
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    return "v_add_f32 v0, v1, v2"

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "; Neural layers map to comment blocks",
      "tier3_extras": "; Extras ignored",
    }
