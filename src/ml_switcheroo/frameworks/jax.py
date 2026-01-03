"""
JAX Core Framework Adapter (Level 0 & Level 1).

This adapter provides support for the functional JAX ecosystem *without* binding
to a high-level neural network library like Flax or Haiku. It maps:
1.  **Level 0 (Core)**: JAX Array API (jnp), Activations (jax.nn), and Types.
2.  **Level 1 (Common Libs)**: Optax (Optimization) and Orbax (Checkpointing).
3.  **IO & Devices**: Handles `save`/`load` via Orbax and `jax.devices` mapping.

It specifically enables `requires_explicit_rng` in plugin traits.
"""

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional, Set

try:
  import jax
  import jax.numpy as jnp
except ImportError:
  jax = None
  jnp = None

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardMap,
  ImportConfig,
  StandardCategory,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("jax")
class JaxCoreAdapter(JAXStackMixin):
  """
  Adapter for Core JAX (jax + optax + orbax) without a Neural Framework.

  Handles translations for:
  -   **Math**: `jnp.abs`, `jnp.sum`, etc.
  -   **Types**: `jnp.float32`, `jnp.int32`, `jnp.bfloat16`.
  -   **Casting**: `.astype(...)` synthesis via plugins.
  -   **Optimization**: `optax.adam`, `optax.sgd`.
  """

  display_name: str = "JAX (Core)"
  inherits_from: Optional[str] = None
  # Set to 10 to ensure it is sorted after Torch (0) but before Numpy (20)
  ui_priority: int = 10

  def __init__(self) -> None:
    """
    Initializes the JAX adapter.
    Detects installation status to toggle between LIVE and GHOST modes.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

    if jax is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("jax")
      if not self._snapshot_data:
        logging.warning("JAX not installed and no snapshot found. Scanning unavailable.")

  # --- Metadata & Imports ---

  @property
  def search_modules(self) -> List[str]:
    """
    Scans only core math and optimization libraries (no neural layers).

    Returns:
        List[str]: List of module names.
    """
    if self._mode == InitMode.GHOST:
      return []
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "optax"]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns a set of submodule names to exclude from recursive introspection.

    Returns:
        Set[str]: Explicitly empty set (safe to scan default paths).
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Defines the canonical import alias ('jax.numpy', 'jnp')."""
    return ("jax.numpy", "jnp")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Self-declared namespace roles.

    Returns:
        Dict[str, ImportConfig]: Map of paths to configuration.
    """
    return {
      "jax": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="jax"),
      "jax.numpy": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="jnp"),
      "optax": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optax"),
      # jax.nn maps to Functional Neural Ops (e.g. activations)
      "jax.nn": ImportConfig(tier=SemanticTier.NEURAL_OPS, recommended_alias="nn"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for identifying API categories.

    Returns:
        Dict[str, List[str]]: Tier to regex patterns mapping.
    """
    return {
      "array": [r"jax\\.numpy\\.", r"jnp\\."],
      "extras": [r"jax\\.random\\.", r"jax\\.lax\\.", r"optax\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """Returns standard JIT-enabled test templates."""
    return self.jax_test_config

  # --- Verification Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """Imports required for JAX initialization logic."""
    return ["import jax", "import jax.random"]

  def get_harness_init_code(self) -> str:
    """
    Returns logic to create JAX PRNG Keys.
    """
    return textwrap.dedent("""
            def _make_jax_key(seed):
                "Attempts to create a JAX PRNGKey."
                try:
                    return jax.random.PRNGKey(seed)
                except (ImportError, AttributeError):
                    return "mock_jax_key"
        """).strip()

  @property
  def declared_magic_args(self) -> List[str]:
    """Returns `key` as a magic state argument."""
    return ["key"]

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines JAX structural behavior (Transformation rules).
    Specifies JIT static arguments for compilation safety.

    Returns:
        StructuralTraits: Configuration object.
    """
    return StructuralTraits(
      module_base=None,
      forward_method="__call__",
      inject_magic_args=[],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      jit_static_args=[
        "axis",
        "axes",
        "dim",
        "dims",
        "keepdim",
        "keepdims",
        "ord",
        "mode",
        "dtype",
      ],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Defines logic capabilities for plugins.
    Enables NumPy compatibility and explicit RNG threading.

    IMPORTANT: Enforces Purity Analysis to catch side-effects unsafe for functional trace.

    Returns:
        PluginTraits: Configuration flags.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
      strict_materialization_method="block_until_ready",
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    """JAX does not use global seeding methods in the imperative sense."""
    return []

  # --- Semantic Definitions (The Spoke) ---

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static Definitions for JAX Core, Optax, Orbax, and Types.
    Loaded dynamically from `frameworks/definitions/jax.json`.

    Returns:
        Dict[str, StandardMap]: Mapping of definitions.
    """
    return load_definitions("jax")

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures for discovering new Standards.
    Supports both Live introspection and Ghost Mode snapshots.

    Args:
        category (StandardCategory): Category to scan.

    Returns:
        List[GhostRef]: Found API references.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """Loads from snapshot."""
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """Scans installed JAX/Optax modules."""
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

    Returns:
        List[GhostRef]: Found activation functions.
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
          found.append(GhostInspector.inspect(obj, f"jax.nn.{name}"))

    except ImportError:
      pass
    return found

  # --- Verification ---

  def convert(self, data: Any) -> Any:
    """
    Converts input data to a JAX array for verification.

    Args:
        data (Any): Input data.

    Returns:
        Any: JAX Array.
    """
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
    Populates the JSON snapshot with manually wired logic.

    Args:
        snapshot (Dict[str, Any]): The snapshot to modify.
    """
    self._apply_stack_wiring(snapshot)

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns a generic JAX example.

    Returns:
        str: Source code.
    """
    return """import jax.numpy as jnp\nfrom jax import grad, jit\n\ndef predict(params, x):\n  return jnp.dot(x, params['w']) + params['b']"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Provides default tiered examples for the base adapter.

    Returns:
        Dict[str, str]: Mapping of tier name to source code.
    """
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "# JAX (Core) does not include a neural network layer library.\n# Use Flax or Haiku for layer abstractions.",
      "tier3_extras": "# Use Optax for optimization:\nimport optax\noptimizer = optax.adam(learning_rate=0.01)",
    }

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Generates JAX core documentation URL.

    Args:
        api_name (str): API path.

    Returns:
        Optional[str]: URL string.
    """
    return super().get_doc_url(api_name)


# Backwards compatibility alias
JaxAdapter = JaxCoreAdapter
