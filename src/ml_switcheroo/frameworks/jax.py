"""
JAX Core Framework Adapter (Level 0 & Level 1).

This adapter provides support for the functional JAX ecosystem *without* binding
to a high-level neural network library like Flax or Haiku. It maps:
1.  **Level 0 (Core)**: JAX Array API (jnp), Activations (jax.nn), and Types.
2.  **Level 1 (Common Libs)**: Optax (Optimization) and Orbax (Checkpointing).

It specifically enables `requires_explicit_rng` in plugin traits.
"""

import logging
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
  InitMode,
  StandardCategory,
  StandardMap,
  ImportConfig,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


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
    """Scans only core math and optimization libraries (no neural layers)."""
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Defines the canonical import alias ('jax.numpy', 'jnp')."""
    return ("jax.numpy", "jnp")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Self-declared namespace roles.
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
    """Regex patterns for identifying API categories."""
    return {"array": [r"jax\\.numpy\\.", r"jnp\\."], "extras": [r"jax\\.random\\.", r"jax\\.lax\\.", r"optax\\."]}

  @property
  def test_config(self) -> Dict[str, str]:
    """Returns standard JIT-enabled test templates."""
    return self.jax_test_config

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines JAX structural behavior (Transformation rules).
    Specifies JIT static arguments for compilation safety.
    """
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
  def plugin_traits(self) -> PluginTraits:
    """
    Defines logic capabilities for plugins.
    Enables NumPy compatibility and explicit RNG threading.

    IMPORTANT: Enforces Purity Analysis to catch side-effects unsafe for functional trace.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
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
    Moved from `standards_internal.py` for decoupling.
    """
    return {
      # --- Math / Array ---
      "TorchFunctional": StandardMap(api="jax.nn"),
      "Abs": StandardMap(api="jnp.abs"),
      "Mean": StandardMap(api="jnp.mean"),
      "Sum": StandardMap(api="jnp.sum"),
      "Add": StandardMap(api="jnp.add"),
      "Sub": StandardMap(api="jnp.subtract"),
      "Mul": StandardMap(api="jnp.multiply"),
      "Div": StandardMap(api="jnp.divide"),
      "Pow": StandardMap(api="jnp.power"),
      "exp": StandardMap(api="jnp.exp"),
      "log": StandardMap(api="jnp.log"),
      "sqrt": StandardMap(api="jnp.sqrt"),
      "square": StandardMap(api="jnp.square"),
      "randn": StandardMap(api="jax.random.normal", requires_plugin="inject_prng"),
      "Clamp": StandardMap(api="jax.numpy.clip", args={"min": "a_min", "max": "a_max", "input": "a"}),
      "Gather": StandardMap(api="jax.numpy.take_along_axis", requires_plugin="gather_adapter"),
      "Scatter": StandardMap(api="jax.ops.index_update", requires_plugin="scatter_indexer"),
      "Flatten": StandardMap(api="jax.numpy.reshape", requires_plugin="flatten_range"),
      "Reshape": StandardMap(api="jax.numpy.reshape", requires_plugin="pack_shape_args"),
      "View": StandardMap(api="jax.numpy.reshape", requires_plugin="view_semantics"),
      "Squeeze": StandardMap(api="jax.numpy.squeeze", args={"dim": "axis"}),
      "Unsqueeze": StandardMap(api="jax.numpy.expand_dims", args={"dim": "axis"}),
      "TopK": StandardMap(api="jax.lax.top_k", requires_plugin="topk_adapter"),
      "ArgMax": StandardMap(api="jax.numpy.argmax", args={"dim": "axis"}),
      "ArgMin": StandardMap(api="jax.numpy.argmin", args={"dim": "axis"}),
      "Pad": StandardMap(api="jax.numpy.pad", requires_plugin="padding_converter"),
      "Einsum": StandardMap(api="jnp.einsum", requires_plugin="einsum_normalizer"),
      "permute_dims": StandardMap(api="jnp.transpose", pack_to_tuple="axes"),
      "size": StandardMap(api="shape", requires_plugin="method_to_property"),
      "OneHot": StandardMap(api="jax.nn.one_hot", args={"tensor": "x", "input": "x"}),
      # --- Activation (Math Tier) ---
      "relu": StandardMap(api="jax.nn.relu"),
      "softmax": StandardMap(api="jax.nn.softmax"),
      "log_softmax": StandardMap(api="jax.nn.log_softmax"),
      "GELU": StandardMap(api="jax.nn.gelu"),
      # --- Optimization (Optax) ---
      "Adam": StandardMap(api="optax.adam", requires_plugin="optimizer_constructor"),
      "SGD": StandardMap(api="optax.sgd", requires_plugin="optimizer_constructor"),
      "RMSprop": StandardMap(api="optax.rmsprop", requires_plugin="optimizer_constructor"),
      "StepLR": StandardMap(api="optax.exponential_decay", requires_plugin="scheduler_rewire"),
      "CosineAnnealingLR": StandardMap(api="optax.cosine_decay_schedule", requires_plugin="scheduler_rewire"),
      "ClipGradNorm": StandardMap(api="optax.clip_by_global_norm", requires_plugin="grad_clipper"),
      "step": StandardMap(api="optimizer_step", requires_plugin="optimizer_step"),
      "zero_grad": StandardMap(api="optimizer_zero_grad", requires_plugin="optimizer_zero_grad"),
      "CrossEntropyLoss": StandardMap(
        api="optax.softmax_cross_entropy_with_integer_labels",
        args={"input": "logits", "target": "labels"},
        requires_plugin="loss_reduction",
      ),
      "MSELoss": StandardMap(
        api="optax.l2_loss", args={"input": "predictions", "target": "targets"}, requires_plugin="loss_reduction"
      ),
      # --- Types (Core Dtypes) ---
      "Float32": StandardMap(api="jnp.float32"),
      "Float64": StandardMap(api="jnp.float64"),
      "Float16": StandardMap(api="jnp.float16"),
      "Int64": StandardMap(api="jnp.int64"),
      "Int32": StandardMap(api="jnp.int32"),
      "Int16": StandardMap(api="jnp.int16"),
      "UInt8": StandardMap(api="jnp.uint8"),
      "Bool": StandardMap(api="jnp.bool_"),
      # --- Casting (via Plugin) ---
      "CastFloat": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastDouble": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastHalf": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastLong": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastInt": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastShort": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastByte": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastBool": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastChar": StandardMap(api="astype", requires_plugin="type_methods"),
      # --- Extras & Contexts ---
      "no_grad": StandardMap(api="contextlib.nullcontext", requires_plugin="context_to_function_wrap"),
      "enable_grad": StandardMap(api="contextlib.nullcontext", requires_plugin="context_to_function_wrap"),
      "DataLoader": StandardMap(api="GenericDataLoader", requires_plugin="convert_dataloader"),
      "LoadStateDict": StandardMap(api="KeyMapper.from_torch", requires_plugin="checkpoint_mapper"),
      # Stub hooks for State Container plugins (used by JAX-based Neural libs via inheritance)
      "register_buffer": StandardMap(api="torch_register_buffer_to_nnx", requires_plugin="torch_register_buffer_to_nnx"),
      "register_parameter": StandardMap(
        api="torch_register_parameter_to_nnx", requires_plugin="torch_register_parameter_to_nnx"
      ),
      "state_dict": StandardMap(api="torch_state_dict_to_nnx", requires_plugin="torch_state_dict_to_nnx"),
      "load_state_dict": StandardMap(api="torch_load_state_dict_to_nnx", requires_plugin="torch_load_state_dict_to_nnx"),
      "parameters": StandardMap(api="torch_parameters_to_nnx", requires_plugin="torch_parameters_to_nnx"),
      # --- Functional Transforms ---
      "vmap": StandardMap(api="jax.vmap", args={"func": "fun"}),
      "grad": StandardMap(api="jax.grad", args={"func": "fun"}),
      "jit": StandardMap(api="jax.jit", args={"func": "fun"}),
    }

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures for discovering new Standards.
    Supports both Live introspection and Ghost Mode snapshots.
    """
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
          found.append(GhostInspector.inspect(obj, f"jax.nn.{name}"))

    except ImportError:
      pass
    return found

  # --- Verification ---

  def convert(self, data: Any) -> Any:
    """
    Converts input data to a JAX array for verification.
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
    """
    self._apply_stack_wiring(snapshot)

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    """Returns a generic JAX example."""
    return """import jax.numpy as jnp\nfrom jax import grad, jit\n\ndef predict(params, x):\n  return jnp.dot(x, params['w']) + params['b']"""

  def get_tiered_examples(self) -> Dict[str, str]:
    """Provides default tiered examples for the base adapter."""
    return {
      "tier1_math": self.get_example_code(),
      "tier2_neural": "# JAX (Core) does not include a neural network layer library.\n# Use Flax or Haiku for layer abstractions.",
      "tier3_extras": "# Use Optax for optimization:\nimport optax\noptimizer = optax.adam(learning_rate=0.01)",
    }


# Backwards compatibility alias
JaxAdapter = JaxCoreAdapter
