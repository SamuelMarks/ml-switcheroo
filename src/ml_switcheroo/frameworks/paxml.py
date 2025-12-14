"""
PaxML (Praxis) Framework Adapter (Level 2).

This adapter specializes the core JAX stack for Google's PaxML framework.
It inherits Level 0 (Core JAX) and Level 1 (Optax/Orbax) capabilities from
`JAXStackMixin` but implements the unique structural traits of the Praxis Layer library.

Key Features:
- **Inheritance**: Uses `JAXStackMixin` for shared math/device/optimization logic.
- **Structural Traits**: Defines `setup` instead of `__init__`, and `__call__` for inference.
- **Layer API**: Maps `praxis.layers` to standard Neural Ops via manual wiring defaults
  to ensure robust bootstrapping even if the environment lacks full Praxis installations.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

# Conditional import for Praxis
try:
  import praxis
  import praxis.layers
  import praxis.base_layer
  import praxis.layers.activations
  import praxis.layers.normalizations
except ImportError:
  praxis = None

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
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


@register_framework("paxml")
class PaxmlAdapter(JAXStackMixin):
  """
  Adapter for PaxML (Praxis Layers) running on JAX.

  This Level 2 adapter leverages the Level 0/1 stack provided by `JAXStackMixin`
  for array operations and optimization, while providing the specific layer
  definitions for the Praxis library.
  """

  display_name: str = "PaxML / Praxis"
  inherits_from: str = "jax"
  ui_priority: int = 60

  def __init__(self):
    """
    Initialize PaxML Adapter.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if praxis is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("paxml")
      if not self._snapshot_data:
        # We log at debug to avoid noise, as static definitions provide fallback
        logging.debug("PaxML (Praxis) not installed and no snapshot found.")

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Scans API surface.
    Delegates Math/Opt to Jax Core, handles Layers specifically for Praxis.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    # Live Scanning
    results = []

    # 1. Reuse Core JAX capabilities (Loss/Optimization/Activations)
    # Praxis often uses Optax or JAX NN directly.
    core = JaxCoreAdapter()
    if category in [StandardCategory.LOSS, StandardCategory.OPTIMIZER, StandardCategory.ACTIVATION]:
      results.extend(core.collect_api(category))

    # 2. Add Praxis Specifics (Layers)
    if category == StandardCategory.LAYER:
      results.extend(self._scan_praxis_layers())

    return results

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """Reads from snapshot."""
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _scan_praxis_layers(self) -> List[GhostRef]:
    """Scans praxis.layers for BaseLayer subclasses."""
    if praxis is None:
      return []

    found = []
    import inspect

    # Scan common submodules
    # Praxis structure is vast, we focus on high-traffic areas
    targets = [praxis.layers]
    # Attempt to import specific submodules if they exist (depend on version)
    try:
      targets.extend([praxis.layers.activations, praxis.layers.normalizations])
    except ImportError:
      pass

    for module in targets:
      for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
          continue

        if inspect.isclass(obj):
          # Heuristic: Check for 'Layer' suffix or inheritance if BaseLayer is available
          # We use name heuristic to avoid importing BaseLayer if not strictly needed
          if "Layer" in name or "Norm" in name or name in ["Linear", "Bias", "StochasticDepth"]:
            try:
              # Use fully qualified path
              api_path = f"{module.__name__}.{name}"
              ref = GhostInspector.inspect(obj, api_path)
              found.append(ref)
            except Exception:
              pass
    return found

  # --- Metadata (Level 2 Specifics) ---

  @property
  def search_modules(self) -> List[str]:
    return [
      "praxis.layers",
      "praxis.base_layer",
      "praxis.layers.activations",
      "optax",  # Inherit visibility of optimizers from stack
    ]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("praxis.layers", "pl")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.praxis\.", r"\.layers\."], "extras": []}

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines PaxML specific structural traits.
    Reflects that Praxis uses a `setup()` method for initialization rather than `__init__`.
    """
    return StructuralTraits(
      module_base="praxis.base_layer.BaseLayer",
      forward_method="__call__",
      init_method_name="setup",
      requires_super_init=False,
      # PaxML handles RNGs via context managers usually, explicit injection is less common
      # in the layer signature itself compared to Flax NNX
      inject_magic_args=[],
      lifecycle_strip_methods=[],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions to ensure core functionality even if discovery fails.
    """
    return {
      # Manually map common layers that might be tricky to discover dynamically
      "Linear": StandardMap(
        api="praxis.layers.Linear", args={"in_features": "input_dims", "out_features": "output_dims"}
      ),
      "Dropout": StandardMap(api="praxis.layers.Dropout", args={"p": "keep_prob"}),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  # --- Converter ---

  def convert(self, data):
    # Delegate to JAX core for array conversion
    return JaxCoreAdapter().convert(data)

  # --- Wiring ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies JAX Stack wiring + PaxML specific logic.
    """
    # 1. Apply Stack Logic (L0 & L1) from Mixin (Math, Optax, Orbax)
    self._apply_stack_wiring(snapshot)

    mappings = snapshot.setdefault("mappings", {})
    imports = snapshot.setdefault("imports", {})

    # 2. Configure Imports
    # Inject standard alias for generated code
    imports["praxis.layers"] = {"root": "praxis", "sub": "layers", "alias": "pl"}

    # 3. Layer specific wiring

    # Ensure Linear layer mapping handles argument discrepancy
    # Standard: in_features, out_features
    # Praxis: input_dims, output_dims
    if "Linear" not in mappings:
      mappings["Linear"] = {"api": "praxis.layers.Linear"}

    if "args" not in mappings["Linear"]:
      mappings["Linear"]["args"] = {}

    mappings["Linear"]["args"]["in_features"] = "input_dims"
    mappings["Linear"]["args"]["out_features"] = "output_dims"
    mappings["Linear"]["args"]["bias"] = "use_bias"

    # Ensure Sequential mapping
    if "Sequential" not in mappings:
      mappings["Sequential"] = {"api": "praxis.layers.Sequential"}

    # Activations often sit in praxis.layers.activations but aliased in praxis.layers
    # or just available via JAX stack (jnp).
    # If discovery found them in praxis, we keep them. If not, stack provides jnp.relu.

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier2_neural": """import praxis.layers as pl
from praxis import base_layer

class SimpleMLP(base_layer.BaseLayer):
  def setup(self):
    # PaxML uses setup() instead of __init__
    self.fc1 = pl.Linear(output_dims=128)
    self.fc2 = pl.Linear(output_dims=10)

  def __call__(self, x):
    x = self.fc1(x)
    x = pl.relu(x)
    return self.fc2(x)
""",
      "tier1_math": JaxCoreAdapter.get_example_code(),
    }
