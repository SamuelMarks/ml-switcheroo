"""
TikZ Framework Adapter.

This module provides the adapter for the TikZ (PGF/TikZ) graphical language.
It registers "tikz" as a valid framework key, allowing the ASTEngine to routing logic
to treat it as a source or target.

Since TikZ is not a Python library, this adapter operates exclusively in GHOST mode,
serving primarily as a metadata container and entry point for the `core.tikz` subsystem.
"""

from typing import List, Tuple, Dict, Any, Set, Optional

from ml_switcheroo.frameworks.base import (
  register_framework,
  InitMode,
  StandardCategory,
  StandardMap,
  ImportConfig,
  GhostRef,
)
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits, OpDefinition
from ml_switcheroo.enums import SemanticTier


@register_framework("tikz")
class TikzAdapter:
  """
  Adapter for TikZ Visualization Language.

  This framework entry enables `ml-switcheroo` to treat visualization code as
  just another "backend". It doesn't provide semantic mappings for math operations
  but serves as the routing endpoint for Graph Extraction/Synthesis.
  """

  display_name: str = "TikZ (LaTeX)"
  inherits_from: Optional[str] = None
  # Place at the very end of visual lists
  ui_priority: int = 1000

  def __init__(self) -> None:
    """
    Initializes the adapter in Ghost Mode.
    """
    self._mode = InitMode.GHOST
    self._snapshot_data: Dict[str, Any] = {}

  @property
  def search_modules(self) -> List[str]:
    """TikZ has no python modules to scan."""
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """No submodules to avoid."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """No imports for TikZ."""
    return ("tikz", "tikz")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """TikZ does not have python namespaces."""
    return {}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """TikZ does not support discovery heuristics."""
    return {}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    TikZ structurally represents Neural networks, though it doesn't execute them.
    """
    return [SemanticTier.NEURAL]

  @property
  def structural_traits(self) -> StructuralTraits:
    """No python structural traits apply to LaTeX."""
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """No plugin traits apply."""
    return PluginTraits()

  @property
  def test_config(self) -> Dict[str, str]:
    """No python tests can be generated for LaTeX."""
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """No harness imports."""
    return []

  def get_harness_init_code(self) -> str:
    """No harness init code."""
    return ""

  def get_to_numpy_code(self) -> str:
    """No runtime conversion logic."""
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> List[str]:
    """No magic args."""
    return []

  @property
  def rng_seed_methods(self) -> List[str]:
    """No RNG methods."""
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """No semantic mappings; translation is structural via Engine pipeline."""
    return {}

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """No specifications defined by this adapter."""
    return {}

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """No API to collect."""
    return []

  def convert(self, data: Any) -> Any:
    """Identity conversion for data."""
    return str(data)

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """No device syntax."""
    return ""

  def get_device_check_syntax(self) -> str:
    """Always False."""
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """No RNG syntax."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """No serialization imports."""
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """No serialization syntax."""
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No wiring to apply."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """No documentation URL for internal DSL."""
    return None

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns a sample TikZ graph.
    """
    return r""" 
% TikZ Example
\begin{tikzpicture} 
    \node [draw] (input) at (0, 0) {Input}; 
    \node [draw] (output) at (0, -2) {Output}; 
    \draw [->] (input) -- (output); 
\end{tikzpicture} 
"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns examples for supported tiers."""
    code = self.get_example_code()
    return {
      "tier1_math": code,
      "tier2_neural": code,
      "tier3_extras": code,
    }
