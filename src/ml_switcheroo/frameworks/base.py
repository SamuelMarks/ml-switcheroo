"""
Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
It acts as the primary contact point between the ``ASTEngine`` and the specific
ML library implementations (plugins).

Key Capabilities (Hybrid Loading):
- Adapters possess an ``init_mode`` (LIVE or GHOST).
- In **LIVE** mode, they introspect installed packages module-by-module.
- In **GHOST** mode, they load cached JSON snapshots.

The ``definitions`` and ``specifications`` properties allow adapters to
self-declare their capabilities and new standards to the SemanticsManager.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Type, Dict, List, Tuple, Optional, Union, Set
from pydantic import BaseModel, Field

from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits, OpDefinition
from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"


class StandardCategory(str, Enum):
  """Enumeration of API categories for discovery."""

  LOSS = "loss"
  OPTIMIZER = "optimizer"
  LAYER = "layer"
  ACTIVATION = "activation"


class InitMode(str, Enum):
  """Initialization mode for adapters."""

  LIVE = "live"
  GHOST = "ghost"


class ImportConfig(BaseModel):
  """
  Configuration for an exposed namespace.

  Used by adapters to self-declare the semantic role of their modules.
  Example: ``torch.nn`` declares itself as ``tier=NEURAL``.
  """

  tier: SemanticTier = Field(description="The semantic category of this namespace.")
  recommended_alias: Optional[str] = Field(default=None, description="Preferred alias (e.g., 'nn').")


class StandardMap(BaseModel):
  """
  Defines how a Framework implements a Middle Layer standard.
  Used for declaring static definitions in adapters.
  """

  api: Optional[str] = Field(
    default=None,
    description="Fully qualified API path in the target framework. If None, implies missing/unsupported functionality.",
  )
  args: Optional[Dict[str, str]] = Field(
    default=None,
    description="Map of {StandardArg: FrameworkArg}. Keys must match the Abstract Standard.",
  )
  inject_args: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
    default=None,
    description="Dictionary of new arguments to inject with fixed literal values.",
  )
  requires_plugin: Optional[str] = Field(
    default=None,
    description="Name of the plugin hook required to handle this operation.",
  )
  output_adapter: Optional[str] = Field(
    default=None,
    description="Lambda string to wrap result (e.g. 'lambda x: x[0]').",
  )
  transformation_type: Optional[str] = Field(
    default=None,
    description="Rewrite mode (e.g. 'infix', 'inline_lambda').",
  )
  operator: Optional[str] = Field(
    default=None,
    description="Infix operator symbol if transformation_type='infix'.",
  )
  pack_to_tuple: Optional[str] = Field(
    default=None,
    description="If set (e.g. 'axes'), collects variadic positional args into a tuple kwargs.",
  )
  macro_template: Optional[str] = Field(
    default=None,
    description="Expression template for composite ops (e.g. '{x} * functional.sigmoid({x})').",
  )
  output_cast: Optional[str] = Field(
    default=None,
    description="Target Dtype string (e.g. 'jnp.int64') to cast the output to via .astype().",
  )
  arg_values: Optional[Dict[str, Dict[str, str]]] = Field(
    default=None,
    description="Map of {StandardArg: {SourceValue: TargetValue}} for enum value translation.",
  )
  required_imports: List[Union[str, Any]] = Field(
    default_factory=list,
    description="List of imports required by this variant.",
  )
  missing_message: Optional[str] = Field(
    default=None,
    description="Custom error message to display if this mapping fails or is explicitly unsupported.",
  )


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.

  Adapters provide translation traits, discovery logic, and syntax generation
  for specific machine learning frameworks.
  """

  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self) -> None:
    """Initialize the adapter."""
    ...

  def convert(self, data: Any) -> Any:
    """
    Converts input data (List, NumPy) to the framework's Tensor format.
    Used by the Fuzzer for validation.

    Args:
       data (Any): Input data in standard format (numpy, list).

    Returns:
       Any: The data converted to the framework's tensor type.
    """
    ...

  @property
  def test_config(self) -> Dict[str, str]:
    """Templates for generating physical test files (imports, conversions)."""
    ...

  @property
  def harness_imports(self) -> List[str]:
    """
    Returns a list of import statements required for the verification harness initialization logic.
    E.g. ["import jax", "import jax.random"]
    """
    ...

  def get_harness_init_code(self) -> str:
    """
    Returns Python code to define initialization helpers for the verification harness.
    Used to inject framework-specific PRNG key generation logic into the generated test script.
    """
    ...

  def get_to_numpy_code(self) -> str:
    """
    Returns a python code snippet that checks if an object 'obj' is a tensor of this framework
    and returns its numpy representation.

    Example:
        return "if hasattr(obj, 'numpy'): return obj.numpy()"
    """
    ...

  @property
  def search_modules(self) -> List[str]:
    """List of module names to scan during discovery."""
    ...

  @property
  def display_name(self) -> str:
    """Human-readable name of the framework."""
    ...

  @property
  def ui_priority(self) -> int:
    """Sort priority for UI display (lower is earlier)."""
    ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Dictionary of Regex patterns for categorizing APIs."""
    ...

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """List of semantic tiers supported by this framework."""
    ...

  @property
  def import_alias(self) -> Tuple[str, str]:
    """The primary import tuple (e.g. ('torch', 'torch'))."""
    ...

  @property
  def inherits_from(self) -> Optional[str]:
    """Optional parent framework key to inherit behavior from."""
    ...

  @property
  def structural_traits(self) -> StructuralTraits:
    """Configuration for class/function rewriting."""
    ...

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Returns flags controlling plugin behavior (e.g. supported casting syntax).
    Default should be PluginTraits().
    """
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """List of global RNG seeding method names."""
    ...

  @property
  def declared_magic_args(self) -> List[str]:
    """
    List of state/context argument names this framework injects/uses
    (e.g., ['rngs'] for Flax, ['key'] for JAX).
    Used to build the global list of arguments to strip from other frameworks.
    """
    ...

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns a set of submodule names to exclude from recursive introspection.
    Prevents crashes on C-extensions or internals (e.g., '_C', 'distributed').
    """
    ...

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Generates code for device object creation.

    Args:
        device_type (str): Type of device (e.g. 'cuda', 'cpu').
        device_index (Optional[str]): Index of device.

    Returns:
        str: Code string.
    """
    ...

  def get_device_check_syntax(self) -> str:
    """
    Returns syntax for checking GPU/Accelerator availability.
    Example: ``torch.cuda.is_available()`` or ``len(jax.devices('gpu')) > 0``.
    """
    ...

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns syntax for splitting RNG state (Preamble Injection).
    Example: ``rng, key = jax.random.split(rng)``.

    Args:
        rng_var (str): Variable name of current RNG state.
        key_var (str): Variable name for new key.
    """
    ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Generates code for save/load operations.

    Args:
        op (str): 'save' or 'load'.
        file_arg (str): File path variable/string.
        object_arg (Optional[str]): Object to save.
    """
    ...

  def get_serialization_imports(self) -> List[str]:
    """Returns list of imports required for serialization."""
    ...

  @classmethod
  def get_example_code(cls) -> str:
    """Returns the primary code example for documentation."""
    ...

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns examples for each supported tier."""
    ...

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Returns the framework's static implementation of Middle Layer Operations."""
    ...

  @property
  def specifications(self) -> Dict[str, OpDefinition]:
    """
    Returns the framework's DEFINITION of operations (The Hub reference).
    Allows adapters to introduce new Abstract Standards to the system.
    """
    ...

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """
    Exposes namespaces provided by this framework with Tier metadata.

    New Self-Declaration Format:
        { "torch.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn") }

    Legacy Format (Deprecated):
        { "source.mod": {"root": "target", "sub": "mod", "alias": "alias"} }
    """
    ...

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects APIs for consensus discovery.

    Args:
        category (StandardCategory): The category to scan (e.g. LOSS, LAYER).

    Returns:
        List[GhostRef]: Found API metadata.
    """
    ...

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual patches to the snapshot dictionary.

    Args:
        snapshot (Dict[str, Any]): The snapshot structure to modify in place.
    """
    ...


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """
  Locates the most recent snapshot for a given framework key.

  Args:
      fw_key: The framework identifier.

  Returns:
      The loaded JSON dictionary or an empty dict if not found.
  """
  if not SNAPSHOT_DIR.exists():
    return {}
  candidates = sorted(list(SNAPSHOT_DIR.glob(f"{fw_key}_v*.json")))
  if not candidates:
    return {}
  target = candidates[-1]
  try:
    with open(target, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception as e:
    logging.error(f"Failed to load snapshot {target}: {e}")
    return {}


_ADAPTER_REGISTRY: Dict[str, Type[FrameworkAdapter]] = {}


def register_framework(name: str):
  """
  Decorator to register a framework adapter class.

  Args:
      name: The unique key for the framework (e.g., 'torch').
  """

  def wrapper(cls):
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """
  Factory to retrieve an instantiated adapter.

  Args:
      name: The framework key.

  Returns:
      An instance of the registered adapter, or None if not found.
  """
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
