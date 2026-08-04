"""TikZ Framework Adapter.

Provides the metadata and configuration required to map intermediate representation
operations and semantics to TikZ/LaTeX visualization output.
"""

from typing import Union, Any, Dict, List, Optional, Tuple
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


@register_framework("tikz")
class TikzAdapter(FrameworkAdapter):
  """Framework adapter for TikZ (LaTeX) visualization.

  This adapter handles structural and semantic translation from intermediate
  representations to TikZ-compatible LaTeX code snippets, enabling the generation
  of visual diagrams representing neural networks and operations.
  """

  display_name: str = "TikZ (LaTeX)"
  inherits_from: Optional[str] = None
  ui_priority: int = 1000
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """Initialize the TikZ framework adapter.

    Sets up the initial state required for TikZ metadata extraction.
    """
    pass

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Get the primary import name and its alias for the framework.

    Returns:
        A tuple of (import_name, alias_name) indicating how the framework
        is imported in generated code.
    """
    return "tikz", "tikz"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Get the sub-namespaces or modules required for framework operations.

    Returns:
        A dictionary mapping sub-namespace identifiers to their corresponding
        import configurations or nested import dictionaries.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Get the test configuration mapping for this adapter.

    Returns:
        A dictionary containing setup, input conversion, and output
        verification templates used during testing of generated snippets.
    """
    return {
      "import": "% latex package imports here",
      "convert_input": "% input {np_var}",
      "to_numpy": "% output {res_var}",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Get the necessary imports required to run test harnesses.

    Returns:
        A list of import statements or package references required by
        the test environment.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Get the initialization code block required for the test harness.

    Returns:
        A string representing the setup or configuration code to execute.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Get the conversion code snippet to transform outputs to NumPy-compatible objects.

    Returns:
        A code snippet as a string that converts objects to Python-native string representations.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the semantic tiers supported by this adapter.

    Returns:
        A list of supported SemanticTier enum values. TikZ adapter
        primarily supports the NEURAL tier for visualizing neural network layers.
    """
    return [SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """Get any magic or framework-specific command-line arguments.

    Returns:
        A list of declared argument names.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Get the structural characteristics and capabilities of the TikZ framework.

    Returns:
        An instance of StructuralTraits containing framework capability configurations.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Get custom extension traits and plug-in configurations for this adapter.

    Returns:
        An instance of PluginTraits detailing registered plugin actions.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Get standard operation mappings and definitions for TikZ.

    Returns:
        A dictionary mapping operation identifiers to StandardMap definitions loaded for TikZ.
    """
    return load_definitions("tikz")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get specific schema definitions and operand constraints for TikZ operations.

    Returns:
        A dictionary mapping operation names to their OperationDef specifications.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Get the methods or functions used to initialize random number generator seeds.

    Returns:
        A list of method names.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Get the syntax required to specify execution devices (e.g., CPU, GPU).

    Args:
        device_type: The type of device (e.g., "cpu", "cuda").
        device_index: The optional index or identifier of the specific device.

    Returns:
        The device selection syntax string (always empty for TikZ).
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """Get the execution syntax to check if device placement is correct or valid.

    Returns:
        A string code fragment that evaluates to True.
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Get the syntax required to split pseudo-random number generator states.

    Args:
        rng_var: The variable holding the current RNG state.
        key_var: The destination variable for the split RNG keys.

    Returns:
        The RNG splitting syntax string (always empty for TikZ).
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Get the import statements required for serializing or saving framework outputs.

    Returns:
        A list of import statements.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Get the framework syntax for serializing an object or output to a file.

    Args:
        op: The serialization operation type or command name.
        file_arg: The destination file path variable.
        object_arg: The optional object variable to be serialized.

    Returns:
        The serialization syntax string.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Get the imports required to convert weights from external models.

    Returns:
        A list of weight conversion module imports.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Get the code snippet to load model weights from a specified path.

    Args:
        path_var: The variable name containing the model weight path.

    Returns:
        A comment string indicating that weights are unsupported in TikZ mode.
    """
    return "# Weights not supported in TikZ mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Get the expression syntax to convert a framework tensor to a NumPy array.

    Args:
        tensor_var: The name of the tensor variable to convert.

    Returns:
        The conversion expression (returns the variable unchanged for TikZ).
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Get the code block to save weight parameters to a file.

    Args:
        state_var: The variable representing the weight parameters state.
        path_var: The variable representing the destination path.

    Returns:
        A comment string indicating that weights are unsupported in TikZ mode.
    """
    return "# Weights not supported in TikZ mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply visual connection wiring or post-processing logic to the representation.

    Args:
        snapshot: A dictionary containing structural snapshot details.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Get the external documentation URL for a given framework API or node.

    Args:
        api_name: The name of the API function or operator.

    Returns:
        The documentation URL string, or None if unavailable.
    """
    return None

  def convert(self, data: Any) -> Any:
    """Convert input data to a format compatible with LaTeX/TikZ serialization.

    Args:
        data: The input raw data or value.

    Returns:
        The converted representation (as a string).
    """
    return str(data)

  def get_tiered_examples(self) -> Dict[str, str]:
    """Get representative examples of TikZ-rendered representations for each semantic tier.

    Returns:
        A dictionary mapping semantic tier identifiers (e.g. "tier1_math",
        "tier2_neural", "tier3_extras") to their corresponding TikZ snippets.
    """
    return {
      "tier1_math": "% Math ops not visualized directly",
      "tier2_neural": "\\begin{tikzpicture}\\node (input) {Input}; \\node (layer) [right of=input] {Layer}; \\draw[->] (input) -- (layer);\\end{tikzpicture}",
      "tier3_extras": "% Extras ignored",
    }
