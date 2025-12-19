"""
Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
It acts as the primary contact point between the `ASTEngine` and the specific
ML library implementations (plugins).

Key Capabilities (Hybrid Loading):
- Adapters possess an `init_mode` (LIVE or GHOST).
- In LIVE mode, they introspect installed packages.
- In GHOST mode, they load cached JSON snapshots.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Type, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier

# Define semantic storage for snapshots relative to this file
# src/ml_switcheroo/frameworks/../snapshots
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"


class StandardCategory(str, Enum):
  """
  Categorization of Framework APIs for automated scanning.
  """

  LOSS = "loss"
  OPTIMIZER = "optimizer"
  LAYER = "layer"
  ACTIVATION = "activation"


class InitMode(str, Enum):
  """
  Operational mode of an adapter.
  LIVE: The framework is installed in the python environment.
  GHOST: The framework is missing; relying on cached snapshots.
  """

  LIVE = "live"
  GHOST = "ghost"


class StandardMap(BaseModel):
  """
  Defines how a Framework implements a Middle Layer standard.
  Used for declaring static definitions in adapters.

  Attributes:
      api (str): Fully qualified API path in the target framework.
      args (Optional[Dict]): Map of {StandardArg: FrameworkArg}.
      requires_plugin (Optional[str]): Name of the plugin hook required for translation.
  """

  api: str = Field(description="Fully qualified API path in the target framework.")
  args: Optional[Dict[str, str]] = Field(
    default=None,
    description="Map of {StandardArg: FrameworkArg}. Keys must match the Abstract Standard.",
  )
  requires_plugin: Optional[str] = Field(
    default=None,
    description="Name of the plugin hook required to handle this operation (e.g., 'decompose_alpha').",
  )


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.

  Implementers must provide properties and methods to support discovery,
  transpilation, and testing. It supports Hybrid Loading via the `_mode`
  attribute (implied convention).
  """

  # We can't strictly enforce instance attributes in Protocol, rely on conventions or properties.
  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self):
    """
    Initialization Logic (Conceptual defaults for implementers).

    Should check if the framework is importable. If not, load snapshot
    via `load_snapshot_for_adapter` and set mode to GHOST.
    """
    ...

  # --- 1. Verification Logic ---
  def convert(self, data: Any) -> Any:
    """
    Converts input data (usually array-like) to this framework's tensor type.

    Used by the Fuzzer to inject valid inputs during verification.

    Args:
        data (Any): Raw input data (list, numpy array).

    Returns:
        Any: The framework-speicific tensor object.
    """
    ...

  # --- 2. Discovery Metadata ---
  @property
  def search_modules(self) -> List[str]:
    """
    Returns the list of python modules to scan when running `ml_switcheroo sync`.

    Returns:
        List[str]: Module names (e.g. ["torch", "torch.nn"]).
    """
    ...

  @property
  def display_name(self) -> str:
    """
    Friendly name for UI/CLI (e.g. 'PyTorch').
    """
    ...

  @property
  def ui_priority(self) -> int:
    """
    Integer defining the sort order in UI tables and Dropdowns.
    Lower numbers appear first.
    """
    ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns used to categorize API surfaces during scaffolding.

    Returns:
        Dict[str, List[str]]: Map of Tier ('neural') -> Regex List ([r'\\.nn\\.']).
    """
    ...

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns the semantic tiers supported by this framework.

    Example:
        NumPy supports [ARRAY_API, EXTRAS] but not NEURAL.

    Returns:
        List[SemanticTier]: Supported tiers.
    """
    ...

  # --- 3. Import Management ---
  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Default module import path and alias for code generation.

    Returns:
        Tuple[str, str]: (module_path, alias) -> `import module_path as alias`.
    """
    ...

  # --- 4. Hierarchy ---
  @property
  def inherits_from(self) -> Optional[str]:
    """
    Key of a parent framework to inherit mappings from.
    Useful for ecosystem overlays (e.g. Flax inherits from JAX).

    Returns:
        Optional[str]: Parent framework key or None.
    """
    ...

  # --- 5. Structural Rewriting (Zero-Edit) ---
  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines how Classes and Functions should be transformed.
    Includes base class names, method renaming rules, and lifecycle stripping.

    Returns:
        StructuralTraits: Configuration object.
    """
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns a list of method names that modify global random state.
    Used by `PurityScanner` to detect side effects.
    """
    ...

  # --- 6. Hardware Abstraction (Zero-Edit) ---
  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns Python source code to instantiate a device on this backend.

    Args:
        device_type (str): 'cuda', 'cpu', etc.
        device_index (Optional[str]): Index string.

    Returns:
        str: Generated python code (e.g. "torch.device('cuda')").
    """
    ...

  # --- 7. IO Serialization (Zero-Edit) ---
  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns Python source code to perform IO operations.

    Args:
        op (str): 'save' or 'load'.
        file_arg (str): Path variable name.
        object_arg (Optional[str]): Object variable name (for save).

    Returns:
        str: Generated python code.
    """
    ...

  def get_serialization_imports(self) -> List[str]:
    """
    Returns a list of import statements required for serialization code to work.

    Returns:
        List[str]: Imports (e.g. ["import torch"]).
    """
    ...

  # --- 8. Documentation & Demo ---
  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns the standard demo example used in the Web Interface.
    """
    ...

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns a dictionary of categorized examples.

    Returns:
        Dict[str, str]: Expected keys: 'tier1_math', 'tier2_neural', 'tier3_extras'.
    """
    ...

  # --- 9. Distributed Semantics (The Middle Layer) ---
  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns the framework's static implementation of Middle Layer Operations.
    Acts as a fallback if dynamic discovery fails.

    Returns:
        Dict[str, StandardMap]: Map of AbstractOp -> Implementation.
    """
    ...

  # --- 10. Ghost Protocol (Introspection) ---
  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Scans the installed framework (LIVE) or reads from snapshot (GHOST).

    Args:
        category (StandardCategory): The semantic category to scan for.

    Returns:
        List[GhostRef]: A list of objects representing the discovered API surface.
    """
    ...

  # --- 11. Manual Wiring (The "Last Mile" Fix) ---
  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies hardcoded semantic wiring rules to the snapshot dictionary.

    This method allows the adapter to inject:
    1. Plugin associations (e.g. requires_plugin="pack_varargs").
    2. Code templates (e.g. fori_loop syntax).
    3. Manual API mappings that automated discovery cannot find.

    Args:
        snapshot (Dict): The mutable dictionary representing the framework's snapshot.
                         (Structure: {"__framework__": "...", "mappings": {...}, "templates": {...}})
    """
    # Default implementation does nothing
    pass


# --- Helper for Implementation Reuse ---


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """
  Locates the most recent snapshot for a given framework key.

  Used by Adapter `__init__` methods when entering Ghost Mode logic.

  Args:
      fw_key (str): The framework identifier (e.g. 'torch').

  Returns:
      Dict[str, Any]: The snapshot data dictionary (containing 'categories', 'version', etc).
                      Returns empty dict if no snapshot found.
  """
  if not SNAPSHOT_DIR.exists():
    logging.warning(f"Snapshot directory missing: {SNAPSHOT_DIR}")
    return {}

  # Find files matching pattern {fw}_v*.json
  # We pick the lexicographically last one assuming it's the latest version
  candidates = sorted(list(SNAPSHOT_DIR.glob(f"{fw_key}_v*.json")))

  if not candidates:
    logging.warning(f"No snapshots found for {fw_key} in {SNAPSHOT_DIR}")
    return {}

  target = candidates[-1]
  logging.info(f"Hydrating {fw_key} from snapshot: {target.name}")

  try:
    with open(target, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception as e:
    logging.error(f"Failed to load snapshot {target}: {e}")
    return {}


# --- Global Registry ---

_ADAPTER_REGISTRY: Dict[str, Type[FrameworkAdapter]] = {}


def register_framework(name: str):
  """
  Decorator to register a framework adapter class.

  Args:
      name (str): The unique key for the framework.
  """

  def wrapper(cls):
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """
  Factory to instantiate an adapter by name.

  Args:
      name (str): The framework key.

  Returns:
      Optional[FrameworkAdapter]: A new instance of the adapter, or None if not found.
  """
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
