"""
JAX Framework Adapter.

Integrates:
- JAX/Flax/Optax Ecosystem.
- Hybrid Discovery (Live + Ghost).
"""

import inspect
import sys
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
  import jax
  import jax.numpy as jnp
except ImportError:
  jax = None
  jnp = None

from .base import (
  register_framework,
  StructuralTraits,
  InitMode,
  StandardCategory,
  StandardMap,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.flax_shim import FlaxScanner


@register_framework("jax")
class JaxAdapter:
  """Adapter for JAX / Flax / Optax."""

  display_name: str = "JAX / Flax"
  inherits_from: Optional[str] = None
  ui_priority: int = 10

  def __init__(self):
    """
    Initialize JAX Adapter.
    Checks for JAX presence. If missing, enters Ghost Mode.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if jax is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("jax")
      if not self._snapshot_data:
        logging.warning("JAX not installed and no snapshot found. Scanning unavailable.")

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """
    Scans the fragmented JAX ecosystem.
    - LOSS/OPTIMIZER -> Optax
    - LAYER -> Flax Linen
    - ACTIVATION -> Flax Linen or JAX NN
    """
    results = []

    if category == StandardCategory.LOSS:
      results.extend(OptaxScanner.scan_losses())

    elif category == StandardCategory.OPTIMIZER:
      results.extend(OptaxScanner.scan_optimizers())

    elif category == StandardCategory.LAYER:
      results.extend(FlaxScanner.scan_layers())

    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_jax_activations())

    return results

  def _scan_jax_activations(self) -> List[GhostRef]:
    """Internal scanner for built-in JAX activations."""
    # Access global jax check
    if jax is None:
      return []
    found = []
    # jax.nn contains relu, silu, etc.
    try:
      # Local import 'as' to prevent shadowing global 'jax' variable in this scope
      import jax.nn as jax_nn

      for name, obj in inspect.getmembers(jax_nn):
        if name.startswith("_"):
          continue

        if inspect.isfunction(obj):
          # Filter common activation names
          if name in ["relu", "gelu", "silu", "elu", "sigmoid", "tanh", "softmax", "log_softmax"]:
            ref = GhostInspector.inspect(obj, f"jax.nn.{name}")
            found.append(ref)
    except ImportError:
      pass
    return found

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "flax.nnx", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("jax.numpy", "jnp")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.linen", r"Module$"], "extras": [r"random\.", r"pmap", r"vmap", r"jit"]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="flax.nnx.Module",
      forward_method="__call__",
      inject_magic_args=[("rngs", "flax.nnx.Rngs")],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "ord", "mode", "dtype"],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  # --- Syntax Generation ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

  def get_serialization_imports(self) -> List[str]:
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"orbax.checkpoint.PyTreeCheckpointer().save(directory={file_arg}, item={object_arg})"
    elif op == "load":
      return f"orbax.checkpoint.PyTreeCheckpointer().restore({file_arg})"
    return ""

  def convert(self, data):
    try:
      import jax.numpy as jnp
    except ImportError:
      return data
    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return jnp.array(data)
    return data
