"""
Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.

Key Updates (Hybrid Loading):
- Adapters now possess an `init_mode` (LIVE or GHOST).
- Added logic to finding and loading cached snapshots if live imports fail.
- Expanded `collect_api` to branch between scanning live objects vs reading cached lists.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Type, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.ghost import GhostRef, GhostInspector

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
  """Operational mode of an adapter."""

  LIVE = "live"
  GHOST = "ghost"


class StandardMap(BaseModel):
  """
  Defines how a Framework implements a Middle Layer standard.
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
  Includes Hybrid Loading Logic via Mixin-like default behavior requires careful subclassing.
  Since Protocol cannot have implementation easily without a base class,
  we define the Interface here and provide a helper method for loading snapshots.
  """

  # We can't strictly enforce instance attributes in Protocol, rely on conventions or properties.
  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self):
    """
    Initialization Logic (Conceptual defaults for implementers):

    try:
        import framework_lib
        self._mode = InitMode.LIVE
    except ImportError:
        self._mode = InitMode.GHOST
        self._load_snapshot()
    """
    ...

  # --- 1. Verification Logic ---
  def convert(self, data: Any) -> Any:
    """Converts input data (usually array-like) to this framework's tensor type."""
    ...

  @classmethod
  def get_import_stmts(cls) -> str:
    """Returns import statements needed for generated test harness."""
    ...

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    """Returns python code to create a tensor from a numpy variable."""
    ...

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    """Returns python code to convert a tensor back to a numpy array."""
    ...

  # --- 2. Discovery Metadata ---
  @property
  def search_modules(self) -> List[str]:
    """List of python modules to scan when running `ml_switcheroo sync`."""
    ...

  @property
  def display_name(self) -> str:
    """Friendly name for UI/CLI (e.g. 'PyTorch')."""
    ...

  @property
  def ui_priority(self) -> int:
    """Integer defining the sort order in UI tables and Dropdowns."""
    ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns used to categorize API surfaces."""
    ...

  # --- 3. Import Management ---
  @property
  def import_alias(self) -> Tuple[str, str]:
    """Default module import path and alias."""
    ...

  # --- 4. Hierarchy ---
  @property
  def inherits_from(self) -> Optional[str]:
    """Key of a parent framework to inherit mappings from."""
    ...

  # --- 5. Structural Rewriting (Zero-Edit) ---
  @property
  def structural_traits(self) -> StructuralTraits:
    """Defines how Classes and Functions should be transformed."""
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """Returns a list of method names that modify global random state."""
    ...

  # --- 6. Hardware Abstraction (Zero-Edit) ---
  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Returns Python source code to instantiate a device on this backend."""
    ...

  # --- 7. IO Serialization (Zero-Edit) ---
  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Returns Python source code to perform IO operations."""
    ...

  def get_serialization_imports(self) -> List[str]:
    """Returns a list of import statements required for serialization to work."""
    ...

  # --- 8. Documentation & Demo ---
  @classmethod
  def get_example_code(cls) -> str:
    """Returns a Python code snippet demonstrating a typical operation."""
    ...

  # --- 9. Distributed Semantics (The Middle Layer) ---
  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Returns the framework's implementation of Middle Layer Operations."""
    ...

  # --- 10. Ghost Protocol (Introspection) ---
  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Scans the installed framework (LIVE) or reads from snapshot (GHOST).

    Args:
        category: The semantic category to scan for.

    Returns:
        A list of GhostRef objects representing the discovered API surface.
    """
    ...


# --- Helper for Implementation Reuse ---


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """
  Locates the most recent snapshot for a given framework key (e.g. 'torch').
  Used by Adapter __init__ methods when entering Ghost Mode.

  Returns:
      The snapshot data dictionary (containing 'categories', 'version', etc).
      Returns empty dict if no snapshot found.
  """
  if not SNAPSHOT_DIR.exists():
    logging.warning(f"Snapshot directory missing: {SNAPSHOT_DIR}")
    return {}

  # Find files matching pattern {fw_key}_v*.json
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
  """Decorator to register a framework adapter class."""

  def wrapper(cls):
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """Factory to instantiate an adapter."""
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
