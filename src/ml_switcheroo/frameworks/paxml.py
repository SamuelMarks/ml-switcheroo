"""
PaxML (Praxis) Framework Adapter (Level 2).

This adapter specializes the core JAX stack for Google's PaxML framework, specifically
targeting the **Praxis** layer library.

It inherits Level 0 (Core JAX) and Level 1 (Optax/Orbax) capabilities from
``JAXStackMixin`` but implements the unique structural traits of the Praxis library,
such as the ``setup()`` lifecycle method for layer definition.
"""

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional, Set

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
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("paxml")
class PaxmlAdapter(JAXStackMixin):
  """
  Adapter for PaxML (Praxis Layers) running on JAX.

  Features:
  -   **Lifecycle Translation**: Maps standard ``__init__`` definitions to Praxis ``setup()`` methods.
  -   **Layer Mapping**: Maps Torch/Flax layers to ``praxis.layers.*``.
  -   **Stack Reuse**: Inherits optimization and math logic from the JAX Core adapter.
  """

  display_name: str = "PaxML / Praxis"
  inherits_from: str = "jax"
  ui_priority: int = 60

  def __init__(self) -> None:
    """
    Initialize PaxML Adapter.

    Checks if ``praxis`` is importable. If not, falls back to Ghost Mode using
    cached snapshots to allow transpilation without installation.
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
    """
    Collects API definitions for the given category.

    Delegates to ``JaxCoreAdapter`` for Math, Loss, and Optimizer categories,
    while handling Layer discovery specifically for Praxis.

    Args:
        category (StandardCategory): The API category to scan.

    Returns:
        List[GhostRef]: Found API signatures.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    results = []
    core = JaxCoreAdapter()
    # Reuse Core JAX discovery for non-layer components
    if category in [StandardCategory.LOSS, StandardCategory.OPTIMIZER, StandardCategory.ACTIVATION]:
      results.extend(core.collect_api(category))

    if category == StandardCategory.LAYER:
      results.extend(self._scan_praxis_layers())
    return results

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """
    Loads API signatures from the JSON snapshot in Ghost Mode.

    Args:
        category (StandardCategory): The category to retrieve.

    Returns:
        List[GhostRef]: Hydrated API references.
    """
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return list(map(GhostInspector.hydrate, raw_list))

  def _scan_praxis_layers(self) -> List[GhostRef]:
    """
    Introspects the live ``praxis.layers`` module.

    Scans ``praxis.layers``, ``activations``, and ``normalizations`` for classes
    inheriting from ``BaseLayer`` or matching naming conventions.

    Returns:
        List[GhostRef]: Discovered layer signatures.
    """
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
          # Check inheritance from BaseLayer if available
          if hasattr(praxis.base_layer, "BaseLayer") and issubclass(obj, praxis.base_layer.BaseLayer):
            is_layer = True
          # Fallback naming heuristic
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
    """
    Returns list of modules to scan during manual scaffolding.

    Returns:
        List[str]: Module names including ``praxis.layers`` and ``praxis.base_layer``.
    """
    if self._mode == InitMode.GHOST:
      return []
    return [
      "praxis.layers",
      "praxis.base_layer",
      "praxis.layers.activations",
      "optax",
    ]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Safe defaults.

    Returns:
        Set[str]: Empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Returns the primary import alias for the framework.

    Returns:
        Tuple[str, str]: ``("praxis.layers", "pl")``.
    """
    return ("praxis.layers", "pl")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Defines the semantic roles of Praxis namespaces.

    Returns:
        Dict[str, ImportConfig]: Mapping of namespaces to tiers.
    """
    return {
      "praxis.layers": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="pl"),
      "praxis.base_layer": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="base_layer"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Returns regex patterns for heuristic categorization.

    Returns:
        Dict[str, List[str]]: Patterns identifying neural components in Praxis.
    """
    return {"neural": [r"\\.praxis\\.", r"\\.layers\\."], "extras": []}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Returns templates for generating physical test files.
    Extends the JAX base config with Praxis imports.

    Returns:
        Dict[str, str]: Code generation templates.
    """
    conf = self.jax_test_config.copy()
    conf["import"] = conf["import"] + "\nimport praxis.layers as pl"
    return conf

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Returns imports required for the verification harness.

    Returns:
        List[str]: ``['import jax', 'import jax.random']``.
    """
    return ["import jax", "import jax.random"]

  def get_harness_init_code(self) -> str:
    """
    Returns Python code helper for initializing JAX random keys in the harness.

    Returns:
        str: Source code for ``_make_jax_key``.
    """
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
    """
    Returns supported semantic tiers.

    Returns:
        List[SemanticTier]: Array API, Neural, and Extras.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    Returns list of magic arguments to strip.
    Praxis usually handles RNG context internally or differently than Flax.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines structural rewriting rules for Praxis.

    Key Differences:
    -   **Module Base**: ``praxis.base_layer.BaseLayer``.
    -   **Init Method**: Replaces ``__init__`` with ``setup``.
    -   **Super Init**: Disabled (not required in Praxis setup).

    Returns:
        StructuralTraits: The configuration object.
    """
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
    """
    Returns plugin capability flags.
    Enables functional control flow and purity analysis (inherited from JAX requirements).

    Returns:
        PluginTraits: The capability flags.
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
    Returns static definitions for Praxis Layers.
    Loaded dynamically from `frameworks/definitions/paxml.json`.

    Returns:
        Dict[str, StandardMap]: The mapping dictionary.
    """
    return load_definitions("paxml")

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns list of global RNG seed methods (Empty for PaxML).

    Returns:
        List[str]: Empty list.
    """
    return []

  def convert(self, data: Any) -> Any:
    """
    Converts input data to JAX arrays.

    Args:
        data (Any): Input data (numpy/list).

    Returns:
        Any: JAX Array.
    """
    return JaxCoreAdapter().convert(data)

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies JAX Stack wiring.

    Injects core JAX math operations and Optax optimizer mappings into the snapshot.

    Args:
        snapshot (Dict[str, Any]): The snapshot dictionary to modify.
    """
    self._apply_stack_wiring(snapshot)

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Generates GitHub search URL for PaxML APIs since documentation is sparse.

    Args:
        api_name: API Path.

    Returns:
        str: URL.
    """
    return f"https://github.com/search?q=repo%3Agoogle%2Fpaxml+{api_name}&type=code"

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns default documentation example code.

    Returns:
        str: Neural Tier (Level 2) example.
    """
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns tiered example code snippets for documentation.

    Returns:
        Dict[str, str]: Mapping of tier IDs to code.
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
