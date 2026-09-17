"""ML-Switcheroo Intermediate Representation (IR) Framework Adapter.

This module provides the framework adapter for the Intermediate Representation
(IR), enabling `ml_switcheroo_ir` to serve as a first-class citizen in the
translation, analysis, and discovery pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier, StandardMap
from ml_switcheroo.frameworks.base import (
  FrameworkAdapter,
  ImportConfig,
  InitMode,
  OperationDef,
  register_framework,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.semantics.schema import PluginTraits, StructuralTraits


@register_framework("ir")
@register_framework("ml_switcheroo_ir")
class IrAdapter(FrameworkAdapter):
  """Adapter for ML-Switcheroo Intermediate Representation (IR).

  This adapter handles metadata, traits, import mappings, and code generation
  hooks for the language-agnostic Intermediate Representation.

  Attributes:
      display_name: Human-readable display label.
      inherits_from: Parent framework identifier if inheriting traits.
      ui_priority: Display priority in user interfaces and CLI tables.
      _mode: Initialization mode (Live vs Ghost).
      _snapshot_data: Cached snapshot metadata mapping.
  """

  display_name: str = "ML-Switcheroo IR (Intermediate Representation)"
  inherits_from: Optional[str] = None
  ui_priority: int = 85
  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[Any, Any] = {}

  def __init__(self) -> None:
    """Initialize the Intermediate Representation framework adapter."""
    pass

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Define the canonical import alias prefix and local name for the IR.

    Returns:
        Tuple[str, str]: Canonical package name and recommended local alias.
    """
    return "ml_switcheroo_ir", "sw_ir"

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Self-declared namespace roles for IR imports.

    Returns:
        Dict[str, Union[Dict[str, str], ImportConfig]]: Mapping of namespace
            paths to their respective import configurations.
    """
    return {
      "ml_switcheroo_ir": ImportConfig(
        tier=SemanticTier.ARRAY_API,
        recommended_alias="sw_ir",
      ),
      "ml_switcheroo_ir.schema": ImportConfig(
        tier=SemanticTier.NEURAL,
        recommended_alias="sw_schema",
      ),
      "ml_switcheroo_ir.types": ImportConfig(
        tier=SemanticTier.EXTRAS,
        recommended_alias="sw_types",
      ),
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """Return standard test config templates for IR.

    Returns:
        Dict[str, str]: Map containing test harness configuration templates.
    """
    return {
      "import": "import ml_switcheroo_ir as sw_ir\\nimport numpy as np",
      "convert_input": "sw_ir.LogicalNode(id='input', kind='Input', metadata={'value': str({np_var})})",
      "to_numpy": "np.asarray({res_var})",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Specify the import list required for the IR test harness.

    Returns:
        List[str]: List of required python import statements.
    """
    return [
      "import ml_switcheroo_ir as sw_ir",
      "from ml_switcheroo_ir import LogicalGraph, LogicalNode",
      "import numpy as np",
    ]

  def get_harness_init_code(self) -> str:
    """Generate the initialization code needed in the test harness.

    Returns:
        str: Test harness initialization code snippet.
    """
    return (
      "import ml_switcheroo_ir as sw_ir\\nfrom ml_switcheroo_ir.validator import Validator\\n_validator = Validator()\\n"
    )

  def get_to_numpy_code(self) -> str:
    """Generate Python code to convert results to NumPy format.

    Returns:
        str: Python code snippet used to perform NumPy conversion.
    """
    return "np.asarray(res)"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Identify the semantic tiers supported by the IR adapter.

    Returns:
        List[SemanticTier]: List of supported SemanticTier enum values.
    """
    return [
      SemanticTier.ARRAY_API,
      SemanticTier.NEURAL,
      SemanticTier.EXTRAS,
    ]

  @property
  def declared_magic_args(self) -> List[str]:
    """List magic or contextual arguments declared by the IR framework.

    Returns:
        List[str]: A list of magic argument name strings.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Return the structural traits configuration for IR.

    Returns:
        StructuralTraits: The structural traits model instance.
    """
    return StructuralTraits(
      module_base="ml_switcheroo_ir.LogicalGraph",
      forward_method="build",
      requires_super_init=False,
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Return the plugin traits configuration for IR.

    Returns:
        PluginTraits: The plugin traits model instance.
    """
    return PluginTraits(
      supports_dynamic_shapes=True,
      supports_sharding=True,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Load and return operation definitions registered for IR.

    Returns:
        Dict[str, StandardMap]: Map of operation keys to standard mapping.
    """
    return load_definitions("ir")

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Get the dictionary of operation specifications for IR.

    Returns:
        Dict[str, OperationDef]: Mapping of operation keys to definition details.
    """
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    """Return methods used to seed random number generation in IR.

    Returns:
        List[str]: A list of method strings.
    """
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate the device syntax for IR device placement.

    Args:
        device_type: Target hardware type (e.g., 'cpu', 'cuda', 'tpu').
        device_index: Optional index pointing to a specific device.

    Returns:
        str: Device placement description string.
    """
    if device_index is not None:
      return f"mesh_axis='{device_type}:{device_index}'"
    return f"mesh_axis='{device_type}'"

  def get_device_check_syntax(self) -> str:
    """Generate device validation/check syntax for IR.

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
        str: IR representation for random splitting.
    """
    return f"{key_var} = sw_ir.split_prng({rng_var})"

  def get_serialization_imports(self) -> List[str]:
    """Define imports needed for model serialization/deserialization.

    Returns:
        List[str]: List of required import strings.
    """
    return ["import json", "import ml_switcheroo_ir as sw_ir"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Provide serialization syntax for loading or saving IR models.

    Args:
        op: Operation type, either 'save' or 'load'.
        file_arg: Target/source file path string.
        object_arg: Optional target object to save.

    Returns:
        str: Generated syntax statement.
    """
    if op == "save":
      return f"Path({file_arg}).write_text({object_arg}.to_json(), encoding='utf-8')"
    return f"sw_ir.LogicalGraph.from_json(Path({file_arg}).read_text(encoding='utf-8'))"

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
        str: Code snippet for loading weights.
    """
    return f"# Weight loading for IR graph from {path_var}"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Construct expression that converts a tensor to a NumPy array.

    Args:
        tensor_var: Variable name of the source tensor.

    Returns:
        str: Conversion expression syntax.
    """
    return f"np.asarray({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Retrieve weight saving code snippet.

    Args:
        state_var: Variable representing the model weight state.
        path_var: Variable representing path to save file.

    Returns:
        str: Code snippet to execute weight saving.
    """
    return f"# Weight saving for IR graph from {state_var} to {path_var}"

  def apply_wiring(self, snapshot: typing.Dict[str, dict]) -> None:
    """Apply a framework wiring snapshot.

    Args:
        snapshot: Dict containing metadata/snapshots to wire.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Retrieve documentation URL for a specific IR API call.

    Args:
        api_name: Name of the API to look up.

    Returns:
        Optional[str]: Documentation URL if available, else None.
    """
    return f"https://github.com/SamuelMarks/ml-switcheroo-ir/blob/main/docs/api/{api_name}.md"

  def convert(
    self, data: typing.Union[int, float, str, list, dict]
  ) -> typing.Union[int, float, str, list, dict, typing.Any]:
    """Convert input data representation to IR compatible format.

    Args:
        data: Arbitrary input data.

    Returns:
        Any: Converted representation.
    """
    if isinstance(data, str):
      try:
        from ml_switcheroo_ir import LogicalGraph

        return LogicalGraph.from_json(data)
      except Exception:
        return str(data)
    return data

  @classmethod
  def get_example_code(cls) -> str:
    """Return a basic example of IR framework code syntax.

    Returns:
        str: Example code block for IR graph.
    """
    return (
      "import ml_switcheroo_ir as sw_ir\n\n"
      "graph = sw_ir.LogicalGraph(\n"
      "    name='SampleModel',\n"
      "    nodes={\n"
      "        'x': sw_ir.LogicalNode(id='x', op_type='Input'),\n"
      "        'relu1': sw_ir.LogicalNode(id='relu1', op_type='Relu', inputs=['x']),\n"
      "    },\n"
      "    outputs=['relu1'],\n"
      ")\n"
    )

  def get_tiered_examples(self) -> Dict[str, str]:
    """Return tiered examples mapped to individual semantic tiers.

    Returns:
        Dict[str, str]: Mapping of tier name to example code string.
    """
    example = self.get_example_code()
    return {
      "tier1_math": example,
      "tier2_neural": example,
      "tier3_extras": example,
    }
