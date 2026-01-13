"""
Base Protocol and Registry for Framework Adapters.

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
  args: Optional[Dict[str, Optional[str]]] = Field(default=None)
  inject_args: Optional[Dict[str, Any]] = Field(default=None)
  requires_plugin: Optional[str] = Field(default=None)
  transformation_type: Optional[str] = Field(default=None)
  operator: Optional[str] = Field(default=None)
  pack_to_tuple: Optional[str] = Field(default=None)
  macro_template: Optional[str] = Field(default=None)
  output_cast: Optional[str] = Field(default=None)
  arg_values: Optional[Dict[str, Dict[str, str]]] = Field(default=None)
  required_imports: List[Union[str, Any]] = Field(default_factory=list)
  missing_message: Optional[str] = Field(default=None)
  layout_map: Optional[Dict[str, str]] = Field(default=None)


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.
  """

  _mode: InitMode = InitMode.LIVE
  _snapshot_data: Dict[str, Any] = {}

  def __init__(self) -> None: ...

  def convert(self, data: Any) -> Any: ...

  @property
  def test_config(self) -> Dict[str, str]: ...

  @property
  def harness_imports(self) -> List[str]: ...

  def get_harness_init_code(self) -> str: ...

  def get_to_numpy_code(self) -> str: ...

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
  def plugin_traits(self) -> PluginTraits: ...

  @property
  def rng_seed_methods(self) -> List[str]: ...

  @property
  def declared_magic_args(self) -> List[str]: ...

  @property
  def unsafe_submodules(self) -> Set[str]: ...

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str: ...

  def get_device_check_syntax(self) -> str: ...

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str: ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str: ...

  def get_serialization_imports(self) -> List[str]: ...

  def get_weight_conversion_imports(self) -> List[str]: ...

  def get_weight_load_code(self, path_var: str) -> str: ...

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str: ...

  def get_weight_save_code(self, state_var: str, path_var: str) -> str: ...

  def get_doc_url(self, api_name: str) -> Optional[str]: ...

  def get_tiered_examples(self) -> Dict[str, str]: ...

  @property
  def definitions(self) -> Dict[str, StandardMap]: ...

  @property
  def specifications(self) -> Dict[str, OperationDef]: ...

  @property
  def import_namespaces(self) -> Dict[str, Union[Dict[str, str], ImportConfig]]: ...

  def collect_api(self, category: StandardCategory) -> List[GhostRef]: ...

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None: ...


def load_snapshot_for_adapter(fw_key: str) -> Dict[str, Any]:
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


def available_frameworks() -> List[str]:
  return list(_ADAPTER_REGISTRY.keys())


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None
