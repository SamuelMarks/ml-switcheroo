"""
JAX Core Framework Adapter (Level 0 & Level 1).

This adapter provides support for the functional JAX ecosystem *without* binding
to a high-level neural network library like Flax or Haiku.

Updates:
- Dynamic scanning of ``jax.nn`` for activations.
- Dynamic scanning of ``optax`` for optimizers/losses via `OptaxScanner`.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

try:
  import jax
  import jax.numpy as jnp
except ImportError:
  jax = None
  jnp = None

from ml_switcheroo.frameworks.base import (
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
    """
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("jax.numpy", "jnp")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"array": [r"jax\\.numpy\\.", r"jnp\\."], "extras": [r"jax\\.random\\.", r"jax\\.lax\\.", r"optax\\."]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base=None,
      forward_method="__call__",
      inject_magic_args=[],
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
    """
    Dynamically scans jax.nn for activation-like functions.
    """
    if jax is None:
      return []
    found = []
    try:
      import jax.nn as jax_nn
      import inspect

      # Iterate everything in jax.nn
      for name, obj in inspect.getmembers(jax_nn):
        if name.startswith("_"):
          continue

        if inspect.isfunction(obj):
          # Heuristic: Does it look like an activation?
          # Most JAX activations are in jax.nn and are lowercase.
          # We filter out known utilities (like 'initializers') if they appear as functions
          # but generally we accept most functions here for consensus voting.
          found.append(GhostInspector.inspect(obj, f"jax.nn.{name}"))

    except ImportError:
      pass
    return found

  # --- Verification ---

  def convert(self, data):
    try:
      import jax.numpy as jnp
    except ImportError:
      return data
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
    return """import jax.numpy as jnp\nfrom jax import grad, jit\n\ndef predict(params, x):\n  return jnp.dot(x, params['w']) + params['b']"""


# Backwards compatibility alias
JaxAdapter = JaxCoreAdapter
