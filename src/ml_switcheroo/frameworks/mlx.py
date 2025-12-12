import numpy as np
from typing import List, Tuple, Optional, Dict
from ml_switcheroo.core.ghost import GhostRef
from .base import register_framework, StructuralTraits, StandardCategory


@register_framework("mlx")
class MLXAdapter:
  """Adapter for Apple MLX."""

  display_name: str = "Apple MLX"
  inherits_from: Optional[str] = None
  ui_priority: int = 50

  @property
  def search_modules(self) -> List[str]:
    return ["mlx.core", "mlx.nn", "mlx.core.fft", "mlx.core.linalg"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("mlx.core", "mx")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.nn\."], "extras": []}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(module_base="mlx.nn.Module", forward_method="__call__", requires_super_init=True)

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    clean_type = device_type.strip("'\"").lower()
    backend_attr = "gpu" if clean_type in ("cuda", "gpu", "mps") else "cpu"

    is_literal = device_type.startswith(("'", '"'))
    type_code = f"mx.{backend_attr}" if is_literal else device_type

    if device_index:
      return f"mx.Device({type_code}, {device_index})"
    return f"mx.Device({type_code})"

  def get_serialization_imports(self) -> List[str]:
    return ["import mlx.core as mx"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"mx.save({file_arg}, {object_arg})"
    elif op == "load":
      return f"mx.load({file_arg})"
    return ""

    # --- Verification ---

  def convert(self, data):
    try:
      import mlx.core as mx
    except ImportError:
      return data
    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return mx.array(data)
    return data
