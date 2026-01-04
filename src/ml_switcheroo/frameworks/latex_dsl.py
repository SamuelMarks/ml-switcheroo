"""
LaTeX Math DSL Adapter.

This framework adapter treats standard mathematical notation (LaTeX) as a target "language"
for documentation purposes. It allows the system to transpile a model architecture
or set of operations into readable mathematical equations.

As a non-computational framework, most runtime hooks (device management,
weight loading, execution) are implemented as no-ops or return LaTeX comments.

Capabilities:
1.  **Format**: Generates LaTeX equation code strings.
2.  **Tiers**: Visual/Mathematical representation only.
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
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.core.latex.parser import LatexParser
from ml_switcheroo.core.latex.emitter import LatexEmitter


@register_framework("latex_dsl")
class LatexDSLAdapter:
  """
  Adapter for generating LaTeX Mathematical Expressions.

  Targeting this framework results in the generation of .tex file content
  describing the model's mathematical formulation (e.g., $y = Wx + b$).
  """

  display_name: str = "LaTeX DSL (MIDL)"
  inherits_from: Optional[str] = None
  ui_priority: int = 98
  _mode: InitMode = InitMode.GHOST

  @property
  def search_modules(self) -> List[str]:
    """
    No modules to search; LaTeX is not a Python library.

    Returns:
        Empty list.
    """
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    No unsafe modules.

    Returns:
        Empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Import alias for the virtual `midl` package used in intermediate AST.

    Returns:
        ("midl", "midl").
    """
    return ("midl", "midl")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Namespace mappings. Defines 'midl' as a valid Neural Tier source.

    Returns:
        Mapping.
    """
    return {
      "midl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="midl"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    No discovery possible.

    Returns:
        Empty dict.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Returns placeholder configuration for testing.

    Returns:
        Comment templates.
    """
    return {
      "import": "% latex package imports (e.g. amsmath)",
      "convert_input": "% input {np_var}",
      "to_numpy": "% output {res_var}",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Returns empty imports.

    Returns:
        Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Returns empty init code.

    Returns:
        Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns generic stringifier.

    Returns:
        Code returning string representation.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Supports Neural and Math tiers for formula generation.

    Returns:
        [NEURAL, ARRAY_API].
    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    No magic args.

    Returns:
        Empty list.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines traits for the virtual 'midl' framework used in round-trip tests.
    When parsing LaTeX, we generate class `M(midl.Module)`.

    Returns:
        Config for parsing LaTeX-generated ASTs.
    """
    return StructuralTraits(
      module_base="midl.Module",
      forward_method="forward",
      init_method_name="__init__",
      requires_super_init=False,
      # No state injection required for math formulas
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Defines default capabilities.

    Returns:
        Default traits.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns static definitions for math mappings.
    Loaded dynamically from `frameworks/definitions/latex_dsl.json`.

    Returns:
        Mappings for equation logic.
    """
    return load_definitions("latex_dsl")

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    No RNG logic.

    Returns:
        Empty list.
    """
    return []

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    No API to collect.

    Args:
        category: Ignored.

    Returns:
        Empty list.
    """
    return []

  # --- Syntax Generation (No-Ops) ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns empty string to avoid polluting latex output with comments about devices.

    Args:
        device_type: Target device.
        device_index: Optional device index.

    Returns:
        Empty string.
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """
    Returns generic boolean.

    Returns:
        "True".
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns empty string.

    Args:
        rng_var: RNG variable name.
        key_var: Key variable name.

    Returns:
        Empty string.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """
    Empty imports.

    Returns:
        Empty list.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns empty string for IO.

    Args:
        op: Operation name.
        file_arg: Filename argument.
        object_arg: Optional object argument.

    Returns:
        Empty string.
    """
    return ""

  # --- Weight Handling (No-Ops) ---

  def get_weight_conversion_imports(self) -> List[str]:
    """
    Empty list.

    Returns:
        Empty list.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """
    Returns comment code.

    Args:
        path_var: Variable name for path.

    Returns:
        Comment code literal.
    """
    return "# Weights not supported in LaTeX mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """
    Identity.

    Args:
        tensor_var: Variable name.

    Returns:
        Argument as is.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """
    Returns comment code.

    Args:
        state_var: Variable name for state.
        path_var: Variable name for path.

    Returns:
        Comment code literal.
    """
    return "# Weights not supported in LaTeX mode"

  # --- Documentation ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    No wiring needed.

    Args:
        snapshot: Snapshot dict.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns None as I haven't documented this yet.

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
        data: Input.

    Returns:
        Stringified input.
    """
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns LaTeX equation example.

    Returns:
        TeX code.
    """
    return r"""\begin{DefModel}{ConvNet}
    \Attribute{conv}{Conv2d}{in=1, out=32, k=3}
    \Return{conv}
\end{DefModel}"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns example TeX strings.

    Returns:
        Examples.
    """
    return {
      "tier1_math": "y = |x| + z",
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "% Extras ignored",
    }

  # --- Engine Factories ---

  def create_parser(self, code: str) -> LatexParser:
    """
    Factory for the ingest parser.

    Args:
        code: Source code string.

    Returns:
        Instance of LatexParser.
    """
    return LatexParser(code)

  def create_emitter(self) -> LatexEmitter:
    """
    Factory for the output generator.

    Returns:
        Instance of LatexEmitter.
    """
    return LatexEmitter()
