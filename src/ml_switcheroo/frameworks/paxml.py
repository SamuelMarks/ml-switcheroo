"""
PaxML (Praxis) Framework Adapter (Level 2).

This adapter specializes the core JAX stack for Google's PaxML framework.
It inherits Level 0 (Core JAX) and Level 1 (Optax/Orbax) capabilities from
``JAXStackMixin`` but implements the unique structural traits of the Praxis library.
"""

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional

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
        logging.debug("PaxML (Praxis) not installed and no snapshot found.")

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    results = []
    core = JaxCoreAdapter()
    if category in [StandardCategory.LOSS, StandardCategory.OPTIMIZER, StandardCategory.ACTIVATION]:
      results.extend(core.collect_api(category))

    if category == StandardCategory.LAYER:
      results.extend(self._scan_praxis_layers())
    return results

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return list(map(GhostInspector.hydrate, raw_list))

  def _scan_praxis_layers(self) -> List[GhostRef]:
    if praxis is None:
      return []

    found = []
    import inspect

    targets = [praxis.layers]
    try:
      targets.extend([praxis.layers.activations, praxis.layers.normalizations])
    except ImportError:
      pass

    for module in targets:
      for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
          continue
        if inspect.isclass(obj):
          is_layer = False
          if hasattr(praxis.base_layer, "BaseLayer") and issubclass(obj, praxis.base_layer.BaseLayer):
            is_layer = True
          elif "Layer" in name or name in ["Linear", "Bias", "StochasticDepth", "Embedding"]:
            is_layer = True

          if is_layer:
            try:
              api_path = f"{module.__name__}.{name}"
              ref = GhostInspector.inspect(obj, api_path)
              found.append(ref)
            except Exception:
              pass
    return found

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    return [
      "praxis.layers",
      "praxis.base_layer",
      "praxis.layers.activations",
      "optax",
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
    conf = self.jax_test_config.copy()
    conf["import"] = conf["import"] + "\nimport praxis.layers as pl"
    return conf

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    return ["import jax", "import jax.random"]

  def get_harness_init_code(self) -> str:
    return textwrap.dedent(""" 
        def _make_jax_key(seed): 
            try: 
                import jax 
                return jax.random.PRNGKey(seed) 
            except ImportError: 
                return "mock_jax_key" 
      """).strip()

  @property
  def supported_tiers(self) -> List[Any]:
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="praxis.base_layer.BaseLayer",
      forward_method="__call__",
      init_method_name="setup",
      requires_super_init=False,
      inject_magic_args=[],
      auto_strip_magic_args=True,
      lifecycle_strip_methods=[],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    return {
      "Linear": StandardMap(
        api="praxis.layers.Linear",
        args={
          "in_features": "input_dims",
          "out_features": "output_dims",
          "bias": "use_bias",
        },
      ),
      "Dropout": StandardMap(api="praxis.layers.Dropout", args={"p": "keep_prob"}),
      "Embedding": StandardMap(api="praxis.layers.Embedding"),
      "Sequential": StandardMap(api="praxis.layers.Sequential"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def convert(self, data: Any) -> Any:
    return JaxCoreAdapter().convert(data)

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    self._apply_stack_wiring(snapshot)

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
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
