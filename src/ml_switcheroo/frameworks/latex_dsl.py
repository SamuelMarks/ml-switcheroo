"""
LaTeX DSL Framework Adapter.

This adapter provides the bridge for the "Machine Intelligence Definition Language" (MIDL).
It enables the engine to consume and produce LaTeX code for neural network architectures.

It exposes:
1.  **Parsing**: Converts MIDL macros into Python AST via `LatexParser`.
2.  **Emission**: Converts Python AST into MIDL macros via `LatexEmitter`.
3.  **Specs**: Defines operations within the `midl` namespace.
"""

from typing import Dict, List, Tuple, Optional, Any, Set

from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
  InitMode,
  GhostRef,
)
from ml_switcheroo.core.latex.parser import LatexParser
from ml_switcheroo.core.latex.emitter import LatexEmitter
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OpDefinition, ParameterDef
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("latex_dsl")
class LatexDSLAdapter(FrameworkAdapter):
  """
  Adapter for the LaTeX DSL (MIDL).
  """

  display_name: str = "LaTeX DSL (MIDL)"
  ui_priority: int = 1000  # Bottom of list
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """Initialize in Ghost Mode as it has no python library backing."""
    self._mode = InitMode.GHOST

  @property
  def search_modules(self) -> List[str]:
    """
    No python modules to scan.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    No unsafe modules.

    Returns:
        Set[str]: Empty.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Virtual namespace for the DSL logic.

    Returns:
        Tuple[str, str]: ("midl", "midl").
    """
    return ("midl", "midl")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Declare `midl` namespace as Neural Logic tier.

    Returns:
        Dict[str, ImportConfig]: Config map.
    """
    return {
      "midl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="midl"),
    }

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Defines how Abstract Standards map to `midl.*` macros.
    Loaded dynamically from `frameworks/definitions/latex_dsl.json`.

    Returns:
        Dict[str, StandardMap]: Definitions.
    """
    return load_definitions("latex_dsl")

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """
    Defines the Abstract Standards themselves if unique to this DSL.
    Currently empty as it reuses standard Neural Ops.

    Returns:
        Dict[str, OpDefinition]: Specs.
    """
    return {}

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines how to parse/emit the Module structure.
    The LatexParser synthesizes classes inheriting from `midl.Module`.

    Returns:
        StructuralTraits: Traits.
    """
    return StructuralTraits(
      module_base="midl.Module",
      forward_method="forward",
      requires_super_init=True,
    )

  # --- Factory Hooks for ASTEngine ---

  def create_parser(self, code: str) -> LatexParser:
    """
    Factory method to instantiate the LaTeX Parser.
    Used by ASTEngine when `source="latex_dsl"`.

    Args:
        code: The input latex source string.

    Returns:
        LatexParser: Configured parser instance.
    """
    return LatexParser(code)

  def create_emitter(self) -> LatexEmitter:
    """
    Factory method to instantiate the LaTeX Emitter.
    Used by ASTEngine when `target="latex_dsl"`.

    Returns:
        LatexEmitter: Configured emitter instance.
    """
    return LatexEmitter()

  # --- Required Protocol Stubs ---

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """No runtime discovery."""
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """No physical tests generated for DSL."""
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """No harness support."""
    return []

  def get_harness_init_code(self) -> str:
    """No harness support."""
    return ""

  def get_to_numpy_code(self) -> str:
    """Identity conversion."""
    return "return data"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Supports Neural ops."""
    return [SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """None."""
    return []

  @property
  def plugin_traits(self) -> PluginTraits:
    """Defaults."""
    return PluginTraits()

  @property
  def rng_seed_methods(self) -> List[str]:
    """None."""
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """No device support."""
    return ""

  def get_device_check_syntax(self) -> str:
    """False."""
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """None."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """None."""
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """None."""
    return ""

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """None."""
    return None

  def convert(self, data: Any) -> Any:
    """Identity."""
    return data

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns example LaTeX source.

    Returns:
        str: Source snippet.
    """
    return r"""\begin{DefModel}{ConvNet}
    \Attribute{conv}{Conv2d}{in=1, out=32, k=3}
    \Attribute{fc}{Linear}{in=..., out=10}
    \Input{x}{[_]}
    \StateOp{h}{conv}{x}{[_]}
    \Op{a}{ReLU}{h}{[_]}
    \StateOp{y}{fc}{a}{[_]}
    \Return{y}
\end{DefModel}"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns map of examples."""
    return {"tier2_neural": self.get_example_code()}

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """No API collection."""
    return []

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No wiring."""
    pass
