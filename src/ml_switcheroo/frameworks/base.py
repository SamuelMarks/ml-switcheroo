"""
Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
It acts as the primary contact point between the `ASTEngine` and the specific
ML library implementations (plugins).

Key Capabilities (Hybrid Loading):
- Adapters possess an `init_mode` (LIVE or GHOST).
- In LIVE mode, they introspect installed packages.
- In GHOST mode, they load cached JSON snapshots.

Update: Enhanced StandardMap schema and import definitions.
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

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"


class StandardCategory(str, Enum):
  LOSS = "loss"
  OPTIMIZER = "optimizer"
  LAYER = "layer"
  ACTIVATION = "activation"


class InitMode(str, Enum):
  LIVE = "live"
  GHOST = "ghost"


class StandardMap(BaseModel):
  """
  Defines how a Framework implements a Middle Layer standard.
  Used for declaring static definitions in adapters.
  """

  api: str = Field(description="Fully qualified API path in the target framework.")
  args: Optional[Dict[str, str]] = Field(
    default=None,
    description="Map of {StandardArg: FrameworkArg}. Keys must match the Abstract Standard.",
  )
  requires_plugin: Optional[str] = Field(
    default=None,
    description="Name of the plugin hook required to handle this operation.",
  )
  # --- Feature 08 Updates: Complex Rewrites ---
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


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.
  """

  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self): ...
  def convert(self, data: Any) -> Any: ...

  @property
  def test_config(self) -> Dict[str, str]: ...

  @property
  def search_modules(self) -> List[str]: ...

  @property
  def display_name(self) -> str: ...

  @property
  def ui_priority(self) -> int: ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]: ...

  @property
  def supported_tiers(self) -> List[SemanticTier]: ...

  @property
  def import_alias(self) -> Tuple[str, str]: ...

  @property
  def inherits_from(self) -> Optional[str]: ...

  @property
  def structural_traits(self) -> StructuralTraits: ...

  @property
  def rng_seed_methods(self) -> List[str]: ...

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str: ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str: ...
  def get_serialization_imports(self) -> List[str]: ...

  @classmethod
  def get_example_code(cls) -> str: ...
  def get_tiered_examples(self) -> Dict[str, str]: ...

  # --- Distributed Semantics ---
  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Returns the framework's static implementation of Middle Layer Operations."""
    ...

  @property
  def import_namespaces(self) -> Dict[str, Dict[str, str]]:
    """
    Returns import path remapping rules.
    Format: {"source.mod": {"root": "target", "sub": "mod", "alias": "alias"}}
    """
    ...

  def collect_api(self, category: StandardCategory) -> List[GhostRef]: ...
  def apply_wiring(self, snapshot: Dict[str, Any]) -> None: ...


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """Locates the most recent snapshot for a given framework key."""
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
  def wrapper(cls):
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
