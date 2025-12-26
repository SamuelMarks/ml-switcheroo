"""
PaxML (Praxis) Framework Adapter (Level 2).

This adapter specializes the core JAX stack for Google's PaxML framework.
It inherits Level 0 (Core JAX) and Level 1 (Optax/Orbax) capabilities from
``JAXStackMixin`` but implements the unique structural traits of the Praxis library.

Refactor: Populates definitions for PaxML layers and namespaces.
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

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  InitMode,
  StandardCategory,
  StandardMap,
  ImportConfig,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.enums import SemanticTier


@register_framework("paxml")
class PaxmlAdapter(JAXStackMixin):
  """
  Adapter for PaxML (Praxis Layers) running on JAX.
  """

  display_name: str = "PaxML / Praxis"
  inherits_from: str = "jax"
  ui_priority: int = 60

  def __init__(self) -> None:
    """
    Initialize PaxML Adapter.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

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
    Delegates Math/Opt to Jax Core, handles Layers (Praxis) dynamically.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    # Live Scanning
    results = []

    # 1. Reuse Core JAX capabilities (Loss/Optimization/Activations)
    # Praxis often uses Optax or JAX NN directly.
    core = JaxCoreAdapter()
    if category in [
      StandardCategory.LOSS,
      StandardCategory.OPTIMIZER,
      StandardCategory.ACTIVATION,
    ]:
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
    return list(map(GhostInspector.hydrate, raw_list))

  def _scan_praxis_layers(self) -> List[GhostRef]:
    """
    Dynamically scans ``praxis.layers`` for BaseLayer subclasses.
    """
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
        # Skip private members
        if name.startswith("_"):
          continue

        if inspect.isclass(obj):
          # Check inheritance from BaseLayer.
          # If direct check fails (import issues), fallback to name heuristic
          is_layer = False
          if hasattr(praxis.base_layer, "BaseLayer") and issubclass(obj, praxis.base_layer.BaseLayer):
            is_layer = True
          # Heuristic fallback ifdirect inheritance check is tricky due to reloading
          elif "Layer" in name or name in ["Linear", "Bias", "StochasticDepth", "Embedding"]:
            is_layer = True

          if is_layer:
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
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    return {
      "praxis.layers": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="pl"),
      "praxis.base_layer": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="base_layer"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\\.praxis\\.", r"\\.layers\\."], "extras": []}

  @property
  def test_config(self) -> Dict[str, str]:
    # Use JAX stack config but add Praxis imports
    conf = self.jax_test_config.copy()
    conf["import"] = conf["import"] + "\nimport praxis.layers as pl"
    return conf

  @property
  def supported_tiers(self) -> List[Any]:
    from ml_switcheroo.enums import SemanticTier

    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

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
      # PaxML handles RNGs via context managers usually, explicit injection is
      # less common in the layer signature itself compared to Flax NNX
      inject_magic_args=[],
      lifecycle_strip_methods=[],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Feature flags for PaxML.
    Inherits JAX functional behavior and purity enforcement.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions to ensure core functionality even if discovery fails.
    """
    return {
      # Manually map common layers that might be tricky to discover dynamically
      "Linear": StandardMap(
        api="praxis.layers.Linear",
        args={
          "in_features": "input_dims",
          "out_features": "output_dims",
          "bias": "use_bias",  # Added Missing Mapping
        },
      ),
      "Dropout": StandardMap(api="praxis.layers.Dropout", args={"p": "keep_prob"}),
      "Embedding": StandardMap(api="praxis.layers.Embedding"),
      "Sequential": StandardMap(api="praxis.layers.Sequential"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  # --- Converter ---

  def convert(self, data: Any) -> Any:
    # Delegate to JAX core for array conversion
    return JaxCoreAdapter().convert(data)

  # --- Wiring ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies JAX Stack wiring + PaxML specific logic.
    """
    # 1. Apply Stack Logic (L0 & L1) from Mixin (Math, Optax, Orbax)
    self._apply_stack_wiring(snapshot)
    # Logic effectively covered by definitions property now, but keeping hooks just in case

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns PaxML (Praxis) idiomatic examples.
    """
    return {
      "tier1_math": JaxCoreAdapter.get_example_code(),
      "tier2_neural": """import praxis.layers as pl
from praxis import base_layer

class SimpleMLP(base_layer.BaseLayer): 
    # Tier 2: Praxis Layer Definition
    # Note the use of 'setup()' instead of '__init__' 
    
    def setup(self): 
        # Layers are instantiated in setup() 
        self.fc1 = pl.Linear(output_dims=128) 
        self.fc2 = pl.Linear(output_dims=10) 
        self.dropout = pl.Dropout(keep_prob=0.5) 

    def __call__(self, x): 
        x = self.fc1(x) 
        x = pl.relu(x) 
        x = self.dropout(x) 
        return self.fc2(x) 
""",
      "tier3_extras": """import praxis.layers as pl
from praxis import pax_fiddle

def get_model_config(): 
    # Tier 3: HParams Configuration
    # Praxis often relies on fiddle/HParams for configuration
    
    p = pax_fiddle.Config(pl.Linear, name="my_linear") 
    p.input_dims = 64
    p.output_dims = 10
    
    return p
""",
    }
