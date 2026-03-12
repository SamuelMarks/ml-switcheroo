"""Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
Updated to remove legacy `create_parser` and `create_emitter` hooks,
enforcing the new Pipeline Routing architecture.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Type, Dict, List, Tuple, Optional, Union, Set
from pydantic import BaseModel, Field

from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits, OperationDef
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
  """Configuration for an exposed namespace."""

  tier: SemanticTier = Field(description="The semantic category of this namespace.")
  recommended_alias: Optional[str] = Field(default=None, description="Preferred alias (e.g., 'nn').")


class StandardMap(BaseModel):
  """Defines how a Framework implements a Middle Layer standard."""

  api: Optional[str] = Field(default=None)
  args: Optional[Dict[str, Optional[Union[str, float, int]]]] = Field(default=None)
  inject_args: Optional[Dict[str, Any]] = Field(default=None)
  requires_plugin: Optional[str] = Field(default=None)
  transformation_type: Optional[str] = Field(default=None)
  operator: Optional[str] = Field(default=None)
  pack_to_tuple: Optional[str] = Field(default=None)
  macro_template: Optional[str] = Field(default=None)
  output_cast: Optional[str] = Field(default=None)
  arg_values: Optional[Dict[str, Union[Dict[str, Any], Any]]] = Field(default=None)
  kwargs_map: Optional[Dict[str, Optional[str]]] = Field(
    default=None, description="Mapping for specific keys within a **kwargs expansion. Values can be null to drop the key."
  )
  required_imports: List[Union[str, Any]] = Field(default_factory=list)
  missing_message: Optional[str] = Field(default=None)
  layout_map: Optional[Dict[str, str]] = Field(default=None)


class FrameworkAdapter(Protocol):
  """Protocol definition for a Framework Adapter."""

  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self) -> None:
    """Execute implementation detail."""
    ...

  def convert(self, data: Any) -> Any:
    """Execute implementation detail."""
    ...

  @property
  def test_config(self) -> Dict[str, str]:
    """Execute implementation detail."""
    ...

  @property
  def harness_imports(self) -> List[str]:
    """Execute implementation detail."""
    ...

  def get_harness_init_code(self) -> str:
    """Execute implementation detail."""
    ...

  def get_to_numpy_code(self) -> str:
    """Execute implementation detail."""
    ...

  @property
  def search_modules(self) -> List[str]:
    """Execute implementation detail."""
    ...

  @property
  def display_name(self) -> str:
    """Execute implementation detail."""
    ...

  @property
  def ui_priority(self) -> int:
    """Execute implementation detail."""
    ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Execute implementation detail."""
    ...

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Execute implementation detail."""
    ...

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Execute implementation detail."""
    ...

  @property
  def inherits_from(self) -> Optional[str]:
    """Execute implementation detail."""
    ...

  @property
  def structural_traits(self) -> StructuralTraits:
    """Execute implementation detail."""
    ...

  @property
  def plugin_traits(self) -> PluginTraits:
    """Execute implementation detail."""
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """Execute implementation detail."""
    ...

  @property
  def declared_magic_args(self) -> List[str]:
    """Execute implementation detail."""
    ...

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Execute implementation detail."""
    ...

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Execute implementation detail."""
    ...

  def get_device_check_syntax(self) -> str:
    """Execute implementation detail."""
    ...

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Execute implementation detail."""
    ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Execute implementation detail."""
    ...

  def get_serialization_imports(self) -> List[str]:
    """Execute implementation detail."""
    ...

  def get_weight_conversion_imports(self) -> List[str]:
    """Execute implementation detail."""
    ...

  def get_weight_load_code(self, path_var: str) -> str:
    """Execute implementation detail."""
    ...

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Execute implementation detail."""
    ...

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Execute implementation detail."""
    ...

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Execute implementation detail."""
    ...

  def get_tiered_examples(self) -> Dict[str, str]:
    """Execute implementation detail."""
    ...

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Execute implementation detail."""
    ...

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    """Execute implementation detail."""
    ...

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]:
    """Execute implementation detail."""
    ...

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """Execute implementation detail."""
    ...

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Execute implementation detail."""
    ...


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
  """Execute implementation detail."""
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
  """Execute implementation detail."""

  def wrapper(cls):
    """Execute implementation detail."""
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def available_frameworks() -> List[str]:
  """Execute implementation detail."""
  return list(_ADAPTER_REGISTRY.keys())


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """Execute implementation detail."""
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
