"""LaTeX Math DSL Adapter.

Provides metadata and hooks for the Machine Intelligence Definition Language (MIDL)
LaTeX DSL.
"""

from typing import Union, Any, Dict, List, Optional, Tuple
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import register_framework, StandardMap, ImportConfig, InitMode, OperationDef
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.core.latex.parser import LatexParser


@register_framework("latex_dsl")
class LatexDSLAdapter:
  """Adapter for generating LaTeX Mathematical Expressions."""

  display_name: str = "LaTeX DSL (MIDL)"
  inherits_from: Optional[str] = None
  ui_priority: int = 98
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """Initialize the LaTeX DSL adapter."""
    pass

  def create_parser(self, code: str) -> LatexParser:
    """Factory for the LaTeX Parser.

    Args:
        code: The string containing the LaTeX mathematical expression or DSL code to parse.

    Returns:
        LatexParser: A parser configured for the provided LaTeX code.
    """
    return LatexParser(code)

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Get the recommended import alias for the LaTeX DSL library.

    Returns:
        Tuple[str, str]: A tuple of (import_name, alias_name).
    """
    return "midl", "midl"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Get the namespace import configuration for MIDL.

    Returns:
        Dict[str, Union[Dict[str, str], ImportConfig]]: A dictionary mapping namespace names
            to their import configurations.
    """
    return {"midl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="midl")}

  @property
  def test_config(self) -> Dict[str, str]:
    """Get the test configuration template for the LaTeX DSL.

    Returns:
        Dict[str, str]: A dictionary containing configuration templates for imports,
            input conversion, and output handling.
    """
    return {"import": "% latex package imports", "convert_input": "% input {np_var}", "to_numpy": "% output {res_var}"}

  @property
  def harness_imports(self) -> List[str]:
    """Get the list of imports needed for test harnesses in LaTeX mode.

    Returns:
        List[str]: An empty list since LaTeX test harness does not require extra library imports.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Get initial setup or boilerplate code for a test harness.

    Returns:
        str: An empty string as no setup code is needed for LaTeX.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Get Python code snippet that converts the output back to a numpy format or string representation.

    Returns:
        str: A Python code snippet that converts the output to a string representation.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the list of semantic tiers supported by the LaTeX DSL.

    Returns:
        List[SemanticTier]: Supported semantic tiers, including NEURAL and ARRAY_API.
    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API]

  @property
  def declared_magic_args(self) -> List[str]:
    """Get any declared magic arguments or parameters handled specially by the DSL.

    Returns:
        List[str]: An empty list as no magic arguments are declared.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Get the structural traits for LaTeX models.

    Returns:
        StructuralTraits: Structural trait definitions containing the module base,
            forward method name, and initialization details.
    """
    return StructuralTraits(
      module_base="midl.Module", forward_method="forward", init_method_name="__init__", requires_super_init=True
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Get plugin-specific traits for the LaTeX DSL adapter.

    Returns:
        PluginTraits: Traits configuring custom plugins for this adapter.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Load and retrieve standard API mappings and definitions for LaTeX DSL operations.

    Returns:
        Dict[str, StandardMap]: Mappings of standard module names (like Conv2d, Linear, Module)
            to their standard definitions.
    """
    defs = load_definitions("latex_dsl")
    if "Module" not in defs:
      defs["Module"] = StandardMap(api="midl.Module")
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="midl.Conv2d", args={"in_channels": "arg_0", "out_channels": "arg_1", "kernel_size": "kernel_size"}
      )
    if "Linear" not in defs:
      defs["Linear"] = StandardMap(api="midl.Linear", args={"in_features": "arg_0", "out_features": "arg_1"})
    return defs

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get detailed specifications of operations supported by the LaTeX DSL.

    Returns:
        Dict[str, OperationDef]: Operation definitions containing parameter metadata
            and descriptions for standard modules like Conv2d and Linear.
    """
    from ml_switcheroo.core.dsl import ParameterDef

    specs = {}
    specs["Conv2d"] = OperationDef(
      operation="Conv2d",
      description="2D Convolution",
      std_args=[ParameterDef(name="in_channels"), ParameterDef(name="out_channels"), ParameterDef(name="kernel_size")],
      variants={},
    )
    specs["Linear"] = OperationDef(
      operation="Linear",
      description="Linear Layer",
      std_args=[ParameterDef(name="in_features"), ParameterDef(name="out_features")],
      variants={},
    )
    return specs

  @property
  def rng_seed_methods(self) -> List[str]:
    """Get names of methods used to seed random number generators.

    Returns:
        List[str]: Empty list as random seeds are not applicable to standard LaTeX math.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Get syntax to place a tensor/model on a device (e.g., cpu, cuda).

    Args:
        device_type: The target device type string.
        device_index: Optional index of the device.

    Returns:
        str: Empty string as device syntax is not supported or needed.
    """
    return ""

  def get_device_check_syntax(self) -> str:
    """Get syntax to verify if a device is active or available.

    Returns:
        str: String representing verification check logic (defaults to "True").
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Get syntax to split random number generator keys.

    Args:
        rng_var: Variable representing the RNG state.
        key_var: Variable to receive the split key.

    Returns:
        str: Empty string as RNG split syntax is not supported or needed.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Get the list of imports required for object serialization.

    Returns:
        List[str]: Empty list as serialization imports are not needed.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Get code or syntax to serialize or deserialize an object.

    Args:
        op: Serialization operation name ("save" or "load").
        file_arg: Target file path or stream.
        object_arg: Optional object reference to save.

    Returns:
        str: Empty string as serialization syntax is not supported.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Get imports required to convert weights or parameters.

    Returns:
        List[str]: Empty list as weight conversion is not supported.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Get code to load weights from a specified path.

    Args:
        path_var: Name of the variable holding the weight file path.

    Returns:
        str: A code snippet indicating that weights are not supported in LaTeX mode.
    """
    return "# Weights not supported in LaTeX mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Get expression to convert a tensor variable to a NumPy array.

    Args:
        tensor_var: Name of the tensor variable.

    Returns:
        str: The unchanged variable string (identity conversion).
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Get code to save model weights.

    Args:
        state_var: Variable holding the model state.
        path_var: Path where weights should be saved.

    Returns:
        str: Comment string indicating weights are not supported in LaTeX mode.
    """
    return "# Weights not supported in LaTeX mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply dynamic mappings or wiring configuration.

    Args:
        snapshot: A dictionary representation of the mappings snapshot.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Get reference documentation URL for a given API symbol.

    Args:
        api_name: Name of the API symbol.

    Returns:
        Optional[str]: None as documentation URLs are not supported for LaTeX DSL.
    """
    return None

  def convert(self, data: Any) -> Any:
    """Convert arbitrary data to its LaTeX DSL or string equivalent.

    Args:
        data: Any input data to convert.

    Returns:
        Any: String representation of the input data.
    """
    return str(data)

  def get_tiered_examples(self) -> Dict[str, str]:
    """Get examples demonstrating code for various semantic tiers.

    Returns:
        Dict[str, str]: A dictionary mapping tier labels to exemplary LaTeX blocks.
    """
    return {
      "tier1_math": "y = |x| + z",
      "tier2_neural": "\\begin{DefModel}{ConvNet} \\Attribute{conv}{Conv2d}{} \\end{DefModel}",
      "tier3_extras": "% Extras ignored",
    }
