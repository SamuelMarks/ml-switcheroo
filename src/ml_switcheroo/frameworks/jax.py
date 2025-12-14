"""
JAX Core Framework Adapter (Level 0 & Level 1).

This adapter provides support for the functional JAX ecosystem *without* binding
to a high-level neural network library like Flax or Haiku. It is intended for:
1.  **Pure Math/Array Conversions** (NumPy <-> JAX Array).
2.  **Functional Optimization** (via Optax mapping).
3.  **Low-Level Checkpointing** (via Orbax).

It implements the `JAXStackMixin` to standardized these common libraries.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

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
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


@register_framework("jax")
class JaxCoreAdapter(JAXStackMixin):
  """
  Adapter for Core JAX (jax + optax + orbax) without a Neural Framework.
  Used for converting math libraries, or as a base for specific frameworks.
  """

  display_name: str = "JAX (Core)"
  inherits_from: Optional[str] = None
  ui_priority: int = 10

  def __init__(self):
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if jax is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("jax")
      if not self._snapshot_data:
        logging.warning("JAX not installed and no snapshot found. Scanning unavailable.")

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    """
    Scans only core math and optimization libraries.
    Does NOT scan flax/haiku modules.
    """
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("jax.numpy", "jnp")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Core JAX API surface patterns."""
    return {"array": [r"jax\.numpy\.", r"jnp\."], "extras": [r"jax\.random\.", r"jax\.lax\.", r"optax\."]}

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines traits for pure JAX code.
    Since pure JAX doesn't use classes for layers, module_base is None.
    """
    return StructuralTraits(
      module_base=None,
      forward_method="__call__",  # Default if user uses classes manually
      inject_magic_args=[],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      # JIT arguments that must be static
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "ord", "mode", "dtype"],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

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
    results = []

    # Level 1: Optax is core to the JAX ecosystem for optimization/loss
    if category == StandardCategory.LOSS:
      results.extend(OptaxScanner.scan_losses())
    elif category == StandardCategory.OPTIMIZER:
      results.extend(OptaxScanner.scan_optimizers())

    # Level 0: JAX Activation functions
    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_jax_activations())

    return results

  def _scan_jax_activations(self) -> List[GhostRef]:
    """Scans jax.nn for activation functions."""
    if jax is None:
      return []
    found = []
    try:
      import jax.nn as jax_nn
      import inspect

      for name, obj in inspect.getmembers(jax_nn):
        if name.startswith("_"):
          continue
        if inspect.isfunction(obj):
          if name in ["relu", "gelu", "silu", "elu", "sigmoid", "tanh", "softmax", "log_softmax"]:
            ref = GhostInspector.inspect(obj, f"jax.nn.{name}")
            found.append(ref)
    except ImportError:
      pass
    return found

  # --- Verification ---

  def convert(self, data):
    try:
      import jax.numpy as jnp
    except ImportError:
      return data
    # Note: In test contexts 'np' usually refers to standard numpy, imported elsewhere if needed for isinstnace checks
    # Assuming input is list/tuple or numpy array
    if hasattr(data, "__array__") or isinstance(data, (list, tuple)):
      return jnp.array(data)
    return data

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies Level 0/1 Stack wiring.
    """
    self._apply_stack_wiring(snapshot)

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    return """import jax.numpy as jnp
from jax import grad, jit

def predict(params, x): 
  # Pure JAX function
  return jnp.dot(x, params['w']) + params['b'] 

def loss_fn(params, x, y): 
  preds = predict(params, x) 
  return jnp.mean((preds - y) ** 2) 
"""


# Backwards compatibility alias for tests importing directly from module
JaxAdapter = JaxCoreAdapter
