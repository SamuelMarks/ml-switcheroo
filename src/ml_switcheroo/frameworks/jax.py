"""JAX Core Framework Adapter (Level 0 & Level 1).

This adapter provides support for the functional JAX ecosystem *without* binding
to a high-level neural network library like Flax or Haiku. It maps:
1.  **Level 0 (Core)**: JAX Array API (jnp), Activations (jax.nn), and Types.
2.  **Level 1 (Common Libs)**: Optax (Optimization) and Orbax (Checkpointing).
3.  **IO & Devices**: Handles `save`/`load` via Orbax and `jax.devices` mapping.

It specifically enables `requires_explicit_rng` in plugin traits.
"""

from ml_switcheroo.enums import RuntimeTensorOrValue

import logging
import textwrap
from types import ModuleType

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.frameworks.base import (
  GhostRef,
  ImportConfig,
  InitMode,
  PluginTraits,
  StandardMap,
  StructuralTraits,
  load_snapshot_for_adapter,
  register_framework,
)
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo.frameworks.loader import load_definitions

jax: ModuleType | None = None
jnp: ModuleType | None = None
try:
  import jax as _jax_module
  import jax.numpy as _jnp_module

  jax = _jax_module
  jnp = _jnp_module
except (ImportError, AttributeError, TypeError, ValueError) as exc:
  logging.getLogger(__name__).debug("Failed to import jax modules: %s", exc)


@register_framework("jax")
class JaxCoreAdapter(JAXStackMixin):
  """Adapter for Core JAX (jax + optax + orbax) without a Neural Framework.

  Handles translations for:
  -   **Math**: `jnp.abs`, `jnp.sum`, etc.
  -   **Types**: `jnp.float32`, `jnp.int32`, `jnp.bfloat16`.
  -   **Casting**: `.astype(...)` synthesis via plugins.
  -   **Optimization**: `optax.adam`, `optax.sgd`.
  """

  display_name: str = "JAX (no framework)"
  inherits_from: str | None = None
  ui_priority: int = 10

  def __init__(self) -> None:
    """Initialize the JAX adapter.

    Detects installation status to toggle between LIVE and GHOST modes.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: dict[str, JsonValue] = {}
    if jax is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("jax")
      if not self._snapshot_data:
        logging.debug("JAX not installed and no snapshot found. Scanning unavailable.")

  @property
  def import_alias(self) -> tuple[str, str]:
    """Define the canonical import alias ('jax.numpy', 'jnp').

    Returns:
        Tuple[str, str]: A tuple containing the canonical module name and its
            recommended import alias (e.g., ("jax.numpy", "jnp")).
    """
    return "jax.numpy", "jnp"

  @property
  def import_namespaces(self) -> dict[str, ImportConfig]:
    """Self-declared namespace roles.

    Returns:
        Dict[str, ImportConfig]: Map of paths to configuration.
    """
    return {
      "jax.numpy": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="jnp"),
      "optax": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optax"),
      "jax.nn": ImportConfig(tier=SemanticTier.NEURAL_OPS, recommended_alias="nn"),
      "flax.nnx": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nnx"),
    }

  @property
  def test_config(self) -> dict[str, str]:
    """Return standard JIT-enabled test templates.

    Returns:
        Dict[str, str]: A dictionary containing standard JIT-enabled test configuration.
    """
    return self.jax_test_config

  @property
  def harness_imports(self) -> list[str]:
    """Import required for JAX initialization logic.

    Returns:
        List[str]: A list of import statement strings required for the harness.
    """
    return ["import jax", "import jax.random"]

  def get_harness_init_code(self) -> str:
    """Return logic to create JAX PRNG Keys.

    Returns:
        str: A string of JAX harness initialization helper code.
    """
    return textwrap.dedent(
      """
            def _make_jax_key(seed):
                "Attempts to create a JAX PRNGKey."
                try:
                    return jax.random.PRNGKey(seed)
                except (ImportError, AttributeError):
                    return "mock_jax_key"
        """
    ).strip()

  @property
  def declared_magic_args(self) -> list[str]:
    """Return `key` as a magic state argument.

    Returns:
        List[str]: A list of magic state argument names.
    """
    return ["key"]

  @property
  def structural_traits(self) -> StructuralTraits:
    """Define JAX structural behavior (Transformation rules).

    Specifies JIT static arguments for compilation safety.

    Returns:
        StructuralTraits: Configuration object.
    """
    return StructuralTraits(
      module_base=None,
      tensor_type="jax.Array",
      forward_method="__call__",
      functional_execution_method="apply",
      inject_magic_args=[],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "ord", "mode", "dtype"],
      default_imports=["from flax import nnx", "import jax.numpy as jnp"],
      layer_namespace_prefix="nnx",
      functional_namespace_prefix="jnp",
      requires_rng_in_init=True,
      sharding_style="with_sharding_constraint",
      module_base_short_prefix=("flax.nnx.", "nnx."),
      fallback_module_base="flax.nnx.Module",
      fallback_magic_injection=("rngs", "nnx.Rngs"),
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Define logic capabilities for plugins.

    Enables NumPy compatibility and explicit RNG threading.

    IMPORTANT: Enforces Purity Analysis to catch side-effects unsafe for functional trace.

    Returns:
        PluginTraits: Configuration flags.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_state=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
      strict_materialization_method="block_until_ready",
      sharding_wrapper_api="jax.experimental.pjit.pjit",
      supports_sharding=True,
    )

  @property
  def rng_seed_methods(self) -> list[str]:
    """JAX does not use global seeding methods in the imperative sense.

    Returns:
        List[str]: Empty list as JAX relies on explicit functional PRNG keys.
    """
    return []

  @property
  def definitions(self) -> dict[str, StandardMap]:
    """Static Definitions for JAX Core, Optax, Orbax, and Types.

    Loaded dynamically from `frameworks/definitions/jax.json`.

    Returns:
        Dict[str, StandardMap]: Mapping of definitions.
    """
    return load_definitions("jax")

  def _collect_ghost(self, category: SemanticTier) -> list[GhostRef]:
    """Load ghost references from the snapshot for the given semantic category.

    Args:
        category (SemanticTier): The semantic category of definitions to collect.

    Returns:
        List[GhostRef]: A list of matching ghost references.
    """
    if not self._snapshot_data:
      return []
    raw_cats = self._snapshot_data.get("categories", {})
    raw_list = raw_cats.get(category.value, []) if isinstance(raw_cats, dict) else []
    assert isinstance(raw_list, list)
    return [GhostRef.model_validate(item) for item in raw_list]

  def _collect_live(self, category: SemanticTier) -> list[GhostRef]:
    """Scan installed JAX/Optax modules to collect live references.

    Args:
        category (SemanticTier): The semantic category of definitions to collect.

    Returns:
        List[GhostRef]: A list of scanned live references.
    """
    results: list[GhostRef] = []
    if category == SemanticTier.LOSS:
      results.extend(getattr(self, "_scan_optax_losses", lambda: [])())
    elif category == SemanticTier.OPTIMIZER:
      results.extend(getattr(self, "_scan_optax_optimizers", lambda: [])())
    elif category == SemanticTier.ACTIVATION:
      results.extend(getattr(self, "_scan_jax_activations", lambda: [])())
    return results

  def convert(self, data: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
    """Convert input data to a JAX array for verification.

    Args:
        data: Input data.

    Returns:
        JAX Array.
    """
    try:
      import jax.numpy as jnp
    except (ImportError, AttributeError) as exc:
      logging.getLogger(__name__).debug("jax.numpy import failed during convert: %s", exc)
      return data
    if hasattr(data, "__array__") or isinstance(data, (list, tuple)):
      try:
        res: RuntimeTensorOrValue = jnp.array(data)
        return res
      except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
        logging.getLogger(__name__).debug("jnp.array conversion failed: %s", exc)
        return data

    return data

  def apply_wiring(self, snapshot: dict[str, dict[str, JsonValue]]) -> None:
    """Apply Level 0/1 Stack wiring.

    Populates the JSON snapshot with manually wired logic.

    Args:
        snapshot (Dict[str, Dict[str, JsonValue]]): The snapshot to modify.
    """
    self._apply_stack_wiring(snapshot)

  def get_tiered_examples(self) -> dict[str, str]:
    """Provide default tiered examples for the base adapter.

    Returns:
        Dict[str, str]: Mapping of tier name to source code.
    """
    return {
      "tier1_math": """import jax.numpy as jnp
from jax import grad, jit

def predict(params, x):
  return jnp.dot(x, params['w']) + params['b']""",
      "tier2_neural": """# JAX (Core) does not include a neural network layer library.
# Use Flax or Haiku for layer abstractions.""",
      "tier4_qwen3-vl": """import jax
import jax.numpy as jnp
import jax.lax as lax

# Pure JAX representation of VisionFrontEnd
class VisionFrontEnd:
    def __init__(self, kernel):
        self.patch_conv_w = kernel

    def __call__(self, x):
        # Vision Patch Embedding Extraction
        return lax.conv_general_dilated(
            lhs=x,
            rhs=self.patch_conv_w,
            window_strides=(2, 14, 14),
            padding='VALID',
            dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
        )
""",
      "tier3_extras": """# Use Optax for optimization:
import optax
optimizer = optax.adam(learning_rate=0.01)""",
    }

  def get_doc_url(self, api_name: str) -> str | None:
    """Generate JAX core documentation URL.

    Args:
        api_name (str): API path.

    Returns:
        Optional[str]: URL string.
    """
    return super().get_doc_url(api_name)
