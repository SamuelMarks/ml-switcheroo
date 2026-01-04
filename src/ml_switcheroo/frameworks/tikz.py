"""
TikZ / LaTeX Visualization Adapter.

This framework adapter treats LaTeX/TikZ as a target "language" for
visualization purposes. It allows the system to transpile a model architecture
into a TikZ node graph representation.

As a non-computational framework, most runtime hooks (device management,
weight loading, execution) are implemented as no-ops or return comments.

Capabilities:
1.  **Format**: Generates LaTeX code strings.
2.  **Tiers**: Visual representation only.
3.  **No-Ops**: Weights, tensors, and runtime checks are explicitly disabled.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
  InitMode,
  OpDefinition,
)
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("tikz")
class TikzAdapter:
  """
  Adapter for generating TikZ (LaTeX) visualizations.

  Targeting this framework results in the generation of a .tex file
  content describing the model's computation graph.
  """

  display_name: str = "TikZ (LaTeX)"
  inherits_from: Optional[str] = None
  ui_priority: int = 1000  # Bottom of list
  _mode: InitMode = InitMode.GHOST

  @property
  def search_modules(self) -> List[str]:
    """
    No modules to search; TikZ is not a python library.

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
    Import alias definition.

    Returns:
        Tuple[str, str]: ("tikz", "tikz").
    """
    return ("tikz", "tikz")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    No namespace mappings.

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
    Returns placeholder configuration for testing.

    Returns:
        Dict[str, str]: Comment templates.
    """
    return {
      "import": "% latex package imports here",
      "convert_input": "% input {np_var}",
      "to_numpy": "% output {res_var}",
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
    Returns generic stringifier.

    Returns:
        str: Code returning string representation.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Supports Neural tier for architecture visualization.

    Returns:
        List[SemanticTier]: [NEURAL].
    """
    return [SemanticTier.NEURAL]

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
    Defines default structural traits (pass-through).

    Returns:
        StructuralTraits: Default traits.
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
    Returns static definitions for visual mappings.
    Loaded dynamically from `frameworks/definitions/tikz.json`.

    Returns:
        Dict[str, StandardMap]: Mappings for drawing logic.
    """
    return load_definitions("tikz")

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
    Returns empty string.

    Args:
        device_type (str): Device.
        device_index (Optional[str]): Index.

    Returns:
        str: Empty string.
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """
    Returns generic boolean.

    Returns:
        str: "True".
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns empty string.

    Args:
        rng_var (str): RNG.
        key_var (str): New Key.

    Returns:
        str: Empty string.
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
    return f"# Weights not supported in TikZ mode"

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
    return f"# Weights not supported in TikZ mode"

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
    Returns None as this TikZ DSL is undocumented.

    Args:
        api_name (str): API name.

    Returns:
        None
    """
    del api_name
    return None

  def convert(self, data: Any) -> Any:
    """
    Returns string representation of data.

    Args:
        data (Any): Input.

    Returns:
        str: Stringified input.
    """
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns TikZ example wrapped in environment.

    Returns:
        str: TeX code.
    """
    return r"\begin{tikzpicture}\node (input) {Input}; \node (layer) [right of=input] {Layer}; \draw[->] (input) -- (layer);\end{tikzpicture}"

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns example TeX strings.

    Returns:
        Dict[str, str]: Examples.
    """
    return {
      "tier1_math": "% Math ops not visualized directly",
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "% Extras ignored",
    }
