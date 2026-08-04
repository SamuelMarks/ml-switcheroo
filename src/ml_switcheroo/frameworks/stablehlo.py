"""StableHLO Framework Adapter.

Provides metadata and hooks for the MLIR/StableHLO stack.
This adapter acts as a metadata container for the Compiler Registry,
identifying StableHLO as a target language and providing static definitions.
"""

from typing import Union, Any, Dict, List, Optional, Tuple
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import ImportConfig, InitMode, StandardMap, register_framework, FrameworkAdapter
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
from ml_switcheroo.core.dsl import OperationDef


@register_framework("stablehlo")
class StableHloAdapter(FrameworkAdapter):
  """Framework adapter for the StableHLO (MLIR) compiler target.

  This adapter implements the FrameworkAdapter interface to provide metadata,
  hooks, and configuration for compiling and executing StableHLO code,
  especially for targets like MLIR and PJRT/XLA.
  """

  display_name: str = "StableHLO (MLIR)"
  inherits_from: Optional[str] = None
  ui_priority: int = 95
  _mode: InitMode = InitMode.LIVE

  def __init__(self) -> None:
    """Initializes the StableHLO adapter.

    Does not require any custom state initialization.
    """
    pass

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Gets the import alias mapping for StableHLO.

    Returns:
        A tuple of (local_name, import_name) for stablehlo.
    """
    return "stablehlo", "stablehlo"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Gets the import namespaces mapping for StableHLO.

    Returns:
        A dictionary mapping namespace strings to their configurations.
    """
    return {}

  @property
  def test_config(self) -> Dict[str, str]:
    """Gets the testing-related configurations for StableHLO.

    Returns:
        A dictionary containing keys like "import", "convert_input", and "to_numpy".
    """
    return {"import": "", "convert_input": "{np_var}", "to_numpy": "np.asarray({res_var})"}

  @property
  def harness_imports(self) -> List[str]:
    """Gets the default list of module imports required by the test harness.

    Returns:
        A list of python import statement strings.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Gets the initialization Python code for the JAX/XLA/PJRT compilation and execution harness.

    Returns:
        A string of executable Python code that sets up XLA compilation, execution of MLIR,
        and conversion of inputs/outputs.
    """
    return """
import jax
from jax.lib import xla_bridge
import numpy as np

_client = xla_bridge.get_backend()

def _execute_mlir(mlir_code: str, *args):
    # PJRT Compilation requires wrapping standard func in a main module if not already
    if "module {" not in mlir_code:
        mlir_code = f"module {{
{mlir_code}
}}"

    executable = _client.compile(mlir_code)
    # Convert numpy inputs to PJRT buffers
    buffers = [_client.buffer_from_pyval(a) for a in args]
    res = executable.execute(buffers)

    if len(res) == 1:
        return res[0]
    return tuple(res)
"""

  def get_to_numpy_code(self) -> str:
    """Gets the code snippet to convert a framework-specific tensor/object to a NumPy array.

    Returns:
        A Python code snippet as a string that converts `obj` to a NumPy array.
    """
    return "return np.asarray(obj)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Gets the semantic tiers supported by the StableHLO framework.

    Returns:
        A list of SemanticTier instances representing the supported tiers.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]

  @property
  def declared_magic_args(self) -> List[str]:
    """Gets any special / magic arguments declared for StableHLO.

    Returns:
        A list of magic argument name strings.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Gets the structural configuration and traits of StableHLO operations.

    Returns:
        A StructuralTraits object containing layout and structural properties.
    """
    return StructuralTraits()

  @property
  def plugin_traits(self) -> PluginTraits:
    """Gets the plugin-specific configuration and traits of StableHLO.

    Returns:
        A PluginTraits object detailing supported hooks and extensions.
    """
    return PluginTraits()

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Loads and returns the standard operation definitions for StableHLO.

    Returns:
        A dictionary mapping operation names to StandardMap configurations.
    """
    return load_definitions("stablehlo")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Gets the inline specifications for StableHLO operations.

    Returns:
        A dictionary mapping operation names to OperationDef specifications.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Gets the names of methods used for seed management in random number generation.

    Returns:
        A list of method name strings.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generates the syntax representing device selection or targeting in StableHLO.

    Args:
        device_type: The type of the target device (e.g., "cpu", "gpu", "tpu").
        device_index: Optional index or ID of the device as a string.

    Returns:
        A string representing the device targeting syntax.
    """
    return f"// Target: {device_type}"

  def get_device_check_syntax(self) -> str:
    """Gets the code or syntax used to check device compatibility or availability.

    Returns:
        A string containing the device check statement or expression.
    """
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generates the syntax or expression for splitting a random number generator key.

    Args:
        rng_var: The name of the random number generator variable.
        key_var: The name of the variable where the split keys will be stored.

    Returns:
        A string snippet for splitting the RNG key.
    """
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Gets any necessary package imports for serialization in StableHLO.

    Returns:
        A list of import statement strings.
    """
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates the syntax needed for serializing StableHLO operations/modules.

    Args:
        op: The serialization operation or method name.
        file_arg: The destination file path variable.
        object_arg: Optional object or module variable to be serialized.

    Returns:
        A string snippet for performing serialization.
    """
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Gets the required imports for converting or processing weights.

    Returns:
        A list of import statement strings.
    """
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    """Generates code for loading weights in the framework.

    Args:
        path_var: The variable or string representing the path to load from.

    Returns:
        A string of code to load weights (comment or fallback since StableHLO doesn't support it directly).
    """
    return "# Weights not supported in StableHLO mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generates the expression to convert a tensor variable to a numpy representation.

    Args:
        tensor_var: The name of the tensor variable.

    Returns:
        A string representing the conversion expression.
    """
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generates code for saving weights from the framework.

    Args:
        state_var: The variable representing the state or weights to save.
        path_var: The variable or string representing the destination path.

    Returns:
        A string of code to save weights (comment or fallback since StableHLO doesn't support it directly).
    """
    return "# Weights not supported in StableHLO mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies wiring modifications or links using snapshot data.

    Args:
        snapshot: A dictionary containing the snapshot mappings and traits.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Gets the official documentation URL for a given StableHLO operation or API name.

    Args:
        api_name: The fully qualified name of the API or operation (e.g., "stablehlo.abs").

    Returns:
        The URL string pointing to the StableHLO specification documentation, or None.
    """
    if api_name.startswith("stablehlo."):
      op_code = api_name.split(".")[-1]
      return f"https://github.com/openxla/stablehlo/blob/main/docs/spec.md#{op_code}"
    return None

  def convert(self, data: Any) -> Any:
    """Converts the input data to a representation suitable for StableHLO (e.g., string representation).

    Args:
        data: The input data or node to be converted.

    Returns:
        The converted representation, typically a string.
    """
    return str(data)

  def get_tiered_examples(self) -> Dict[str, str]:
    """Gets standard examples of code snippets for StableHLO across different semantic tiers.

    Returns:
        A dictionary mapping tier names to code example strings.
    """
    return {
      "tier1_math": "%0 = stablehlo.abs %arg0 : tensor<*xf32>",
      "tier2_neural": "module { func.func @main() { %0 = stablehlo.convolution(...) } }",
      "tier3_extras": "// Extras ignored",
    }
