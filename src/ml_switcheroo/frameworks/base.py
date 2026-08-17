"""Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
Updated to remove legacy `create_parser` and `create_emitter` hooks,
enforcing the new Pipeline Routing architecture.
"""

from typing import Any

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Protocol, Type, Dict, List, Tuple, Optional, Union
from pydantic import BaseModel, Field
from ml_switcheroo_ir.schema.ghost import SemanticTier, StandardMap, GhostRef
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
from ml_switcheroo.core.dsl import OperationDef as OperationDef

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "ml-compiler-snapshots"


class InitMode(str, Enum):
  """Initialization mode for adapters."""

  LIVE = "live"
  GHOST = "ghost"


class ImportConfig(BaseModel):
  """Configuration for an exposed namespace."""

  tier: SemanticTier = Field(description="The semantic category of this namespace.")
  recommended_alias: Optional[str] = Field(default=None, description="Preferred alias (e.g., 'nn').")


class FrameworkAdapter(Protocol):
  """Protocol definition for a Framework Adapter.

  Defines the comprehensive set of properties, configuration options,
  and code generation methods required to integrate a machine learning framework
  into the translation and analysis pipelines.
  """

  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self) -> None:
    """Initialize the framework adapter instance."""
    ...

  def convert(self, data: Any) -> Any:
    """Convert framework-specific or generic data structures.

    Args:
        data: The input framework-specific or raw data structure to be converted.

    Returns:
        The converted representation compatible with the pipeline.
    """
    ...

  @property
  def test_config(self) -> Dict[str, str]:
    """Retrieve testing-specific configurations for the adapter.

    Returns:
        A dictionary of test configurations, such as framework-specific
        test flags, tolerances, or environment overrides.
    """
    ...

  @property
  def harness_imports(self) -> List[str]:
    """Get the necessary module import statements for generating a test harness.

    Returns:
        A list of import strings (e.g., ["import torch", "import torch.nn as nn"]).
    """
    ...

  def get_harness_init_code(self) -> str:
    """Generate framework-specific initialization code for the test harness.

    Returns:
        A string containing Python code to initialize the framework environment
        (e.g., setting global states or precision).
    """
    ...

  def get_to_numpy_code(self) -> str:
    """Retrieve the Python code snippet used to convert a framework tensor to a NumPy array.

    Returns:
        A code string defining helper logic to convert variables to NumPy arrays.
    """
    ...

  @property
  def display_name(self) -> str:
    """Return the user-facing display name of the framework.

    Returns:
        The display name of the framework (e.g., "PyTorch", "JAX").
    """
    ...

  @property
  def ui_priority(self) -> int:
    """Return the sorting or rendering priority of this framework in UI views.

    Returns:
        An integer indicating priority. Higher values represent higher priority.
    """
    ...

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Get the semantic tiers supported by this framework adapter.

    Returns:
        A list of supported SemanticTier values (e.g., TENSOR, NEURAL_NETWORK).
    """
    ...

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Return the standard import name and recommended package alias for this framework.

    Returns:
        A tuple of (module_name, recommended_alias), e.g., ("torch", "torch").
    """
    ...

  @property
  def inherits_from(self) -> Optional[str]:
    """Identify if this framework adapter inherits behavior from another adapter.

    Returns:
        The name of the parent framework if one exists, otherwise None.
    """
    ...

  @property
  def structural_traits(self) -> StructuralTraits:
    """Retrieve structural traits and preferences of the framework.

    Returns:
        The structural traits configuration of the framework, dictating
        syntactic constraints like keyword-only arguments or native array layouts.
    """
    ...

  @property
  def plugin_traits(self) -> PluginTraits:
    """Retrieve plugin-specific traits and capabilities supported by the framework.

    Returns:
        The plugin traits configuration highlighting enabled transform plugins.
    """
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """List the framework APIs or methods used to set random number generator seeds.

    Returns:
        A list of seeding method names or statements.
    """
    ...

  @property
  def declared_magic_args(self) -> List[str]:
    """Return special or magic argument names that require translation overrides.

    Returns:
        A list of argument names that require custom mapping logic (e.g., 'device', 'dtype').
    """
    ...

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate framework-specific syntax for device placement.

    Args:
        device_type: The device type target (e.g., "cuda", "cpu", "mps").
        device_index: Optional string representing the specific device ID or index.

    Returns:
        A string representing the device instantiation/placement syntax.
    """
    ...

  def get_device_check_syntax(self) -> str:
    """Generate framework-specific code for querying/verifying available devices.

    Returns:
        A Python code snippet that checks and returns device status or availability.
    """
    ...

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generate the syntax for splitting random number generator states/keys.

    Args:
        rng_var: Variable name of the source RNG state or key.
        key_var: Variable name where the split keys will be assigned.

    Returns:
        A code snippet representing the RNG split operation.
    """
    ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generate framework-specific syntax for serializing or deserializing models/weights.

    Args:
        op: Serialization operation type, typically "save" or "load".
        file_arg: File path variable or string representation.
        object_arg: Optional variable name of the object to be saved/serialized.

    Returns:
        A Python code snippet executing the serialization action.
    """
    ...

  def get_serialization_imports(self) -> List[str]:
    """Get the necessary module imports for model/weight serialization operations.

    Returns:
        A list of required import statement strings.
    """
    ...

  def get_weight_conversion_imports(self) -> List[str]:
    """Get the necessary module imports for weight format conversion.

    Returns:
        A list of required import statement strings.
    """
    ...

  def get_weight_load_code(self, path_var: str) -> str:
    """Generate framework-specific code to load weights from a file path.

    Args:
        path_var: Variable containing the source weight file path.

    Returns:
        A code snippet to load the weights into the framework's state dictionary/model.
    """
    ...

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generate an expression converting a framework tensor/array to a NumPy array.

    Args:
        tensor_var: Variable name of the framework tensor.

    Returns:
        The expression code string (e.g., "tensor_var.detach().cpu().numpy()").
    """
    ...

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generate framework-specific code to save weights/states to a file path.

    Args:
        state_var: Variable containing model weights or state dictionary.
        path_var: Variable containing target destination file path.

    Returns:
        A code snippet to save the weights/state.
    """
    ...

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Retrieve the official documentation URL for a given framework API.

    Args:
        api_name: The name or path of the framework API.

    Returns:
        The documentation URL string if found, otherwise None.
    """
    ...

  def get_tiered_examples(self) -> Dict[str, str]:
    """Provide standard usage examples for different semantic tiers.

    Returns:
        A dictionary mapping semantic tier or category names to example code snippets.
    """
    ...

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Return the semantic mapping definitions of framework APIs to the IR standard map.

    Returns:
        A dictionary mapping framework API names to their StandardMap definitions.
    """
    ...

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Retrieve the full operation signatures and specifications for framework APIs.

    Returns:
        A dictionary mapping operation names to their OperationDef specifications.
    """
    ...

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Return the configured namespaces and import rules exposed by this framework.

    Returns:
        A dictionary mapping namespace identifiers to import configurations or maps.
    """
    ...

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply dynamic bindings or overrides onto the adapter using snapshot data.

    Args:
        snapshot: A dictionary containing API binding snapshot data.
    """
    ...


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """Load the most recent snapshot JSON data for a given framework key from the SNAPSHOT_DIR.

  Args:
      fw_key: The identifier key representing the framework adapter.

  Returns:
      The parsed JSON snapshot dictionary, or an empty dictionary if no snapshots
      are found or if loading fails.
  """
  if not SNAPSHOT_DIR.exists():
    return {}
  candidates = sorted(list(SNAPSHOT_DIR.glob(f"{fw_key}_v*.json")))
  if not candidates:
    return {}
  target = candidates[-1]
  try:
    with open(target, "r", encoding="utf-8") as f:
      return json.load(f)  # type: ignore
  except Exception as e:
    logging.error(f"Failed to load snapshot {target}: {e}")
    return {}


_ADAPTER_REGISTRY: Dict[str, Type[FrameworkAdapter]] = {}


def register_framework(name: str) -> Any:
  """Get a decorator to register concrete FrameworkAdapter classes under a specific name.

  Args:
      name: The global registration key/name for the framework.

  Returns:
      A decorator function that maps the decorated class to the given name
      in the registry and returns the class itself.
  """

  def wrapper(cls) -> Any:  # type: ignore
    """Register the decorated class in the framework adapter registry.

    Args:
        cls: The FrameworkAdapter implementation class.

    Returns:
        The unchanged adapter class.
    """
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def available_frameworks() -> List[str]:
  """Retrieve the names of all currently registered framework adapters.

  Returns:
      A list of registered framework adapter keys/names.
  """
  return list(_ADAPTER_REGISTRY.keys())


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """Instantiate and return the registered FrameworkAdapter subclass corresponding to the given name.

  Args:
      name: The registered name of the framework adapter.

  Returns:
      An instance of FrameworkAdapter if registered, otherwise None.
  """
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None


__all__ = ["SemanticTier", "StandardMap", "GhostRef", "FrameworkAdapter", "PluginTraits", "StructuralTraits"]
