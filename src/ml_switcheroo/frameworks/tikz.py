"""
TikZ Framework Adapter.

This module provides the adapter for the TikZ (LaTeX) visualization target.
It primarily serves as a target for the Graph Extraction -> Latex Emitter pipeline.
Unlike computational frameworks, it has few semantic definitions, but it
adheres to the adapter protocol for consistency.
"""

from typing import List, Tuple, Dict, Any, Optional, Set

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
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("tikz")
class TikzAdapter(FrameworkAdapter):
  """
  Adapter for TikZ (LaTeX Visualization).
  """

  display_name: str = "TikZ (LaTeX)"
  ui_priority: int = 1000  # Bottom of list
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """Initialize in Ghost Mode as it has no python library backing."""
    self._mode = InitMode.GHOST

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns static definitions for TikZ.
    Loaded dynamically from `frameworks/definitions/tikz.json`.

    Returns:
        Dict[str, StandardMap]: Definitions map.
    """
    return load_definitions("tikz")

  @property
  def specifications(self) -> Dict:
    """
    Returns abstract specifications.

    Returns:
        Dict: Empty dict.
    """
    return {}

  @property
  def search_modules(self) -> List[str]:
    """
    Returns list of modules to scan.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns unsafe submodules.

    Returns:
        Set[str]: Empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Returns import alias.

    Returns:
        Tuple[str, str]: ("tikz", "tikz").
    """
    return ("tikz", "tikz")

  @property
  def import_namespaces(self) -> Dict:
    """
    Returns namespaces.

    Returns:
        Dict: Empty dict.
    """
    return {}

  @property
  def discovery_heuristics(self) -> Dict:
    """
    Returns regex heuristics.

    Returns:
        Dict: Empty dict.
    """
    return {}

  @property
  def test_config(self) -> Dict:
    """
    Returns test templates.

    Returns:
        Dict: Empty dict.
    """
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """
    Returns harness imports.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Returns harness init code.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns numpy conversion logic.

    Returns:
        str: Identity logic.
    """
    return "return data"

  def convert(self, data: Any) -> Any:
    """
    Converts input data for validation.

    Args:
        data (Any): Input.

    Returns:
        Any: String representation.
    """
    return str(data)

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns supported tiers.

    Returns:
        List[SemanticTier]: [NEURAL].
    """
    return [SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    Returns magic args.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Returns structural traits.

    Returns:
        StructuralTraits: Default object.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Returns plugin capabilities.

    Returns:
        PluginTraits: Default object.
    """
    return PluginTraits()

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns seed methods.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns device syntax.

    Args:
        device_type: Device.
        device_index: Index.

    Returns:
        str: Empty string.
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """
    Returns check syntax.

    Returns:
        str: "False".
    """
    return "False"

  def get_rng_split_syntax(self, r, k) -> str:
    """
    Returns split syntax.

    Args:
        r: RNG var.
        k: Key var.

    Returns:
        str: Empty string.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """
    Returns IO imports.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_serialization_syntax(self, op: str, f: str, o: Optional[str] = None) -> str:
    """
    Returns IO syntax.

    Args:
        op: Operation.
        f: File.
        o: Object.

    Returns:
        str: Empty string.
    """
    return ""

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns doc URL.

    Args:
        api_name: API.

    Returns:
        Optional[str]: None.
    """
    return None

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns example code.

    Returns:
        str: LaTeX snippet.
    """
    return r"\begin{tikzpicture} ... \end{tikzpicture}"

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns tiered examples.

    Returns:
        Dict[str, str]: Examples map.
    """
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": self.get_example_code(),
      "tier3_extras": self.get_example_code(),
    }

  def collect_api(self, c) -> List:
    """
    Collects API signatures.

    Args:
        c: Category.

    Returns:
        List: Empty list.
    """
    return []

  def apply_wiring(self, s) -> None:
    """
    Applies wiring.

    Args:
        s: Snapshot.
    """
    pass
