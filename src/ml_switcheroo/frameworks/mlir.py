"""MLIR Framework Adapter.

Simplified to only provide Metadata.
"""

import typing


from typing import Union, Dict, List, Optional, Tuple
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


@register_framework("mlir")
class MlirAdapter(FrameworkAdapter):
  """Adapter for MLIR.

  This adapter handles metadata configurations and standard properties
  for the MLIR framework.
  """

  display_name: str = "MLIR (Intermediate)"
  inherits_from: Optional[str] = None
  ui_priority: int = 90
  _mode: InitMode = InitMode.GHOST

  def __init__(self) -> None:
    """Initialize the MLIR framework adapter.

    Since MLIR is metadata-only in this adapter, this constructor performs
    no operations.
    """
    pass

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Define the canonical import alias prefix and name for MLIR.

    Returns:
        Tuple[str, str]: A tuple containing the module name and its recommended
            import alias.
    """
    return "mlir", "sw"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Self-declared namespace roles for MLIR imports.

    Returns:
        Dict[str, Union[Dict[str, str], ImportConfig]]: Mapping of paths to
            their respective import configurations.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Return standard test config templates for MLIR.

    Returns:
        Dict[str, str]: Map containing the test harness configuration templates
            such as imports and conversions.
    """
    return {
      "import": "// module attributes",
      "convert_input": "// input tensor {np_var}",
      "to_numpy": "// result tensor {res_var}",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Specify the import list required for the MLIR test harness.

    Returns:
        List[str]: List of required python import statements.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Generate the initialization code needed in the test harness.

    Returns:
        str: Test harness initialization code snippet.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Generate Python code to convert results to NumPy format.

    Returns:
        str: The Python code snippet used to perform the NumPy conversion.
    """
    return "return str(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Identify the semantic tiers supported by the MLIR adapter.

    Returns:
        List[SemanticTier]: List of supported SemanticTier enum values.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """List magic or contextual arguments declared by this framework.

    Returns:
        List[str]: A list of magic argument name strings.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Return the structural traits configuration for MLIR.

    Returns:
        StructuralTraits: The structural traits model instance.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Return the plugin traits configuration for MLIR.

    Returns:
        PluginTraits: The plugin traits model instance.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Load and returns operation definitions registered for MLIR.

    Returns:
        Dict[str, StandardMap]: Map of operation keys to their standard mapping.
    """
    return load_definitions("mlir")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get the dictionary of operation specifications for MLIR.

    Returns:
        Dict[str, OperationDef]: Mapping of operation keys to definition details.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Return methods used to seed random number generation in MLIR.

    Returns:
        List[str]: A list of method strings.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate the device syntax for MLIR device placement.

    Args:
        device_type: The target hardware type (e.g., "cpu", "gpu", "tpu").
        device_index: Optional index pointing to a specific device.

    Returns:
        str: The target device syntax statement.
    """
    return f"// Target: {device_type}"

  def get_device_check_syntax(self) -> str:
    """Generate device validation/check syntax for MLIR.

    Returns:
        str: Python-compatible code snippet checking device status.
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generate the syntax needed for splitting RNG state.

    Args:
        rng_var: Variable name for the input random state.
        key_var: Target variable name for the output key.

    Returns:
        str: MLIR representation for random splitting.
    """
    return f"// Split RNG: {rng_var} -> {key_var}"

  def get_serialization_imports(self) -> List[str]:
    """Define imports needed for model serialization/deserialization.

    Returns:
        List[str]: List of required import strings.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Provide serialization syntax for loading or saving MLIR models.

    Args:
        op: Operation type, either "save" or "load".
        file_arg: Target/source file path string.
        object_arg: Optional target object to save.

    Returns:
        str: The generated syntax statement.
    """
    if op == "save":
      return f"// Save {object_arg} to {file_arg}"
    return f"// Load from {file_arg}"

  def get_weight_conversion_imports(self) -> List[str]:
    """Retrieve imports needed for converting weights.

    Returns:
        List[str]: List of import statements.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Retrieve weight loading code snippet.

    Args:
        path_var: Variable name containing path to weight file.

    Returns:
        str: Python/MLIR code snippet for loading weights.
    """
    return "# Weights loading not supported in MLIR adapter"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Construct expression that converts a tensor to a NumPy array.

    Args:
        tensor_var: Variable name of the source tensor.

    Returns:
        str: Conversion expression syntax.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Retrieve weight saving code snippet.

    Args:
        state_var: Variable representing the model weight state.
        path_var: Variable representing path to save file.

    Returns:
        str: Code snippet to execute weight saving.
    """
    return "# Weights saving not supported in MLIR adapter"

  def apply_wiring(self, snapshot: typing.Dict[str, dict]) -> None:
    """Apply a framework wiring snapshot.

    Args:
        snapshot: A dict containing metadata/snapshots to wire.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Retrieve documentation URL for a specific MLIR API call.

    Args:
        api_name: Name of the API to look up.

    Returns:
        Optional[str]: Documentation URL if available, else None.
    """
    return None

  def convert(
    self, data: typing.Union[int, float, str, list, dict]
  ) -> typing.Union[int, float, str, list, dict, typing.Any]:
    """Convert input data representation to MLIR compatible format.

    Args:
        data: Arbitrary input data.

    Returns:
        Any: Converted representation.
    """
    return str(data)

  @classmethod
  def get_example_code(cls) -> str:
    """Return a basic example of MLIR framework code syntax.

    Returns:
        str: Example code block for MLIR module.
    """
    return """// Example MLIR
sw.module {
^entry:
    sw.func {sym_name = "main"} {
        %0 = sw.op(%x) {type = "torch.abs"}
    }
}"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """Return tiered examples mapped to individual semantic tiers.

    Returns:
        Dict[str, str]: Mapping of tier name to example code string.
    """
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "// Extras ignored",
    }
