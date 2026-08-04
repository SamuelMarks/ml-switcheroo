"""HTML/SVG DSL Framework Adapter.

This module provides the HtmlDSLAdapter class, which implements the
FrameworkAdapter protocol for the HTML/SVG Visual DSL. It simplifies the
representation of the HTML Grid DSL, providing standard metadata, traits,
and parsing capabilities without executing or compiling code.
"""

from typing import Union, Dict, List, Tuple, Any, Optional
from ml_switcheroo.frameworks.base import register_framework, FrameworkAdapter, StandardMap, ImportConfig, InitMode
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.core.html.parser import HtmlParser


@register_framework("html")
class HtmlDSLAdapter(FrameworkAdapter):
  """Adapter for the HTML/SVG Visual DSL.

  This adapter supports the SemanticTier.NEURAL tier, allowing HTML-based
  visual block descriptions to be mapped and structured. It defines specific
  structural traits (e.g., using `html_dsl.Module` as the base class) and
  implements metadata retrieval for standard operators like Module and Conv2d.
  """

  display_name: str = "HTML Grid DSL"
  ui_priority: int = 980
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """Initialize the HTML DSL Framework Adapter.

    Sets the initialization mode to InitMode.GHOST and prepares an empty
    snapshot data dictionary.
    """
    self._mode = InitMode.GHOST
    self._snapshot_data: Dict[str, Any] = {}

  def create_parser(self, code: str) -> HtmlParser:
    """Factory for creating an HTML Parser used by Ingestion.

    Args:
      code: The HTML visual DSL code string to be parsed.

    Returns:
      An instance of HtmlParser configured with the provided HTML code.
    """
    return HtmlParser(code)

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Retrieve the default import alias for this framework.

    Returns:
      A tuple containing the target module name and its recommended import alias
      (e.g., ("html_dsl", "dsl")).
    """
    return "html_dsl", "dsl"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Get the namespace import configurations for this framework.

    Returns:
      A dictionary mapping namespace names to their ImportConfig settings or
      alias sub-dictionaries.
    """
    return {"html_dsl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="dsl")}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the semantic tiers supported by this framework.

    Returns:
      A list of supported SemanticTier values (e.g., [SemanticTier.NEURAL]).
    """
    return [SemanticTier.NEURAL]

  @property
  def structural_traits(self) -> StructuralTraits:
    """Get structural traits of modules/layers within this framework.

    Returns:
      A StructuralTraits object specifying the module base class, forward method,
      initializer name, and whether a parent init call is required.
    """
    return StructuralTraits(
      module_base="html_dsl.Module", forward_method="forward", init_method_name="__init__", requires_super_init=True
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Get custom plugin traits required by this framework.

    Returns:
      An empty PluginTraits object configuration.
    """
    return PluginTraits()

  @property
  def test_config(self) -> Dict[str, str]:
    """Get the test configuration parameters for this adapter.

    Returns:
      An empty dictionary containing the test configuration options.
    """
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """Get additional imports required for the test harness.

    Returns:
      An empty list of Python import statements or module names.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Generate initialization code required for the test harness.

    Returns:
      An empty string representing standard initialization statements.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Generate python code used to convert framework objects to NumPy arrays.

    Returns:
      A Python code snippet as a string that converts objects to standard
      string representations.
    """
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> List[str]:
    """Get the list of declared magic arguments handled by this framework.

    Returns:
      An empty list of argument name strings.
    """
    return []

  @property
  def rng_seed_methods(self) -> List[str]:
    """Get the names of methods used for random number generator seeding.

    Returns:
      An empty list of API names.
    """
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Retrieve standard operator and layer mapping definitions.

    Returns:
      A dictionary mapping layer names (e.g., 'Module', 'Conv2d') to their
      StandardMap definitions.
    """
    defs = load_definitions("html_dsl")
    if "Module" not in defs:
      defs["Module"] = StandardMap(api="html_dsl.Module")
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="html_dsl.Conv2d", args={"in_channels": "i", "out_channels": "o", "kernel_size": "k"}
      )
    return defs

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Retrieve specifications for custom DSL operators.

    Returns:
      An empty dictionary mapping operator names to their OperationDef specifications.
    """
    return {}

  def convert(self, data: Any) -> Any:
    """Convert external data/tensors into the framework's native format.

    Args:
      data: The input data to convert.

    Returns:
      The converted data, which is represented as a string in HTML mode.
    """
    return str(data)

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate syntax for targeting specific hardware devices.

    Args:
      device_type: The target device type (e.g., 'cpu', 'gpu').
      device_index: Optional index or identifier of the specific device.

    Returns:
      An empty string representing the device selection syntax.
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """Generate syntax to check the active device of a tensor.

    Returns:
      A string containing the expression 'False' to indicate device assignment is unsupported.
    """
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generate syntax for splitting random number generator states.

    Args:
      rng_var: Variable holding the parent RNG state.
      key_var: Name of the variable to store the split sub-key.

    Returns:
      An empty string containing the split statement.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Get imports required for serialization routines.

    Returns:
      An empty list of import statement strings.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generate syntax for serialization or deserialization of models/states.

    Args:
      op: The operation name (e.g., 'load', 'save').
      file_arg: Path or file-like object argument.
      object_arg: Optional name of the object variable to serialize.

    Returns:
      An empty string containing the target statement.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Get imports needed for weight conversion processes.

    Returns:
      An empty list of import statements.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Generate code snippet for loading weights.

    Args:
      path_var: String variable referencing the path to the weight file.

    Returns:
      A string snippet indicating weights are not supported in HTML mode.
    """
    return "# Weights not supported in HTML mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generate syntax for converting a tensor variable to a NumPy array.

    Args:
      tensor_var: Name of the tensor variable.

    Returns:
      A string expression representing the conversion logic.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generate code snippet for saving model weights.

    Args:
      state_var: Name of the state/weight dictionary variable.
      path_var: Variable referencing the destination file path.

    Returns:
      A string snippet indicating weights are not supported in HTML mode.
    """
    return "# Weights not supported in HTML mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply snapshot parameter wiring to resolve framework structures.

    Args:
      snapshot: A dictionary containing snapshot state configuration.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Retrieve documentation URLs for standard operations/APIs.

    Args:
      api_name: Name of the API or operator.

    Returns:
      None, as HTML documentation URLs are not supported.
    """
    return None

  def get_tiered_examples(self) -> Dict[str, str]:
    """Retrieve example snippets categorized by semantic tier.

    Returns:
      A dictionary mapping tier names to illustrative visual code blocks.
    """
    return {
      "tier2_neural": """
<div class="grid">
  <div class="box r">
    <span class="header-txt">conv: Conv2d</span>
    <code>i=1, o=32, k=3</code>
  </div>
</div>
"""
    }
