"""
Flax NNX Framework Adapter (Level 2).

This adapter builds upon the core JAX stack to support the **Flax NNX**
neural network library. It inherits math/optimizer logic from ``JAXStackMixin``
but implements dynamic discovery for NNX Modules.

It explicitly selects the `repack_attn_flax` strategy for Attention layers.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
  import flax.nnx
except ImportError:
  flax_nnx = None

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  InitMode,
  ImportConfig,
  StandardCategory,
  StandardMap,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin

# Import core adapter for shared validation/conversion logic
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


@register_framework("flax_nnx")
class FlaxNNXAdapter(JAXStackMixin):
  """
  Adapter for the Flax NNX Framework (The Object-Oriented JAX API).

  Links standard Neural Layer definitions to ``flax.nnx.*``.
  """

  display_name: str = "Flax NNX"
  inherits_from: str = "jax"
  ui_priority: int = 15

  def __init__(self) -> None:
    """Initializes adapter, checking if Flax is installed."""
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

    try:
      import flax.nnx

      self._flax_available = True
    except ImportError:
      self._flax_available = False
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("flax_nnx")
      if not self._snapshot_data:
        logging.warning("Flax NNX not installed and no snapshot found.")

  # --- Discovery (Dynamic Introspection) ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API definitions.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    # Live Scanning
    results = []

    # 1. Use Core JAX capabilities (Optimization/Loss/Activations)
    # We instantiate a temporary core adapter to reuse its scanning logic
    core = JaxCoreAdapter()
    if category in [
      StandardCategory.LOSS,
      StandardCategory.OPTIMIZER,
      StandardCategory.ACTIVATION,
    ]:
      results.extend(core.collect_api(category))

    # 2. Add Flax Specifics (Layers)
    if category == StandardCategory.LAYER:
      results.extend(self._scan_nnx_layers())

    return results

  def _scan_nnx_layers(self) -> List[GhostRef]:
    """
    Dynamically scans ``flax.nnx`` for Module subclasses.
    """
    if not self._flax_available:
      return []

    found = []
    try:
      from flax import nnx
      import inspect

      for name, obj in inspect.getmembers(nnx):
        if name.startswith("_"):
          continue

        if inspect.isclass(obj):
          if issubclass(obj, nnx.Module) and name != "Module":
            found.append(GhostInspector.inspect(obj, f"flax.nnx.{name}"))

    except Exception as e:
      logging.debug(f"Error scanning flax.nnx: {e}")

    return found

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    return ["jax.numpy", "flax.nnx", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("flax.nnx", "nnx")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Self-declared namespace roles for Flax NNX.
    Inherits JAX imports via delegation logic in Manager, but exposes its own here.
    """
    # NOTE: Order matters for hydration overwrite!
    # Ensure flax.nnx is processed LAST so it wins the NEURAL tier provider slot.
    # This prevents 'from flax import linen as nn' when 'nnx' is desired.
    return {
      "flax.linen": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "flax.nnx": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nnx"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\\.nnx\\.", r"\\.linen\\."], "extras": [r"random\\."]}

  @property
  def test_config(self) -> Dict[str, str]:
    conf = self.jax_test_config.copy()
    conf["import"] = conf["import"] + "\nimport flax.nnx as nnx"
    return conf

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="nnx.Module",
      forward_method="__call__",
      inject_magic_args=[("rngs", "nnx.Rngs")],
      requires_super_init=False,
      jit_static_args=[
        "axis",
        "axes",
        "dim",
        "dims",
        "keepdim",
        "keepdims",
        "dtype",
      ],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Capabilities of Flax NNX.
    Opt-in to Purity Analysis as it runs on JAX.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_state=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for Flax NNX.
    Only defines Neural Layers; delegates Math/Opt to Jax Core.
    """
    return {
      "Linear": StandardMap(
        api="flax.nnx.Linear",
        args={"in_features": "in_features", "out_features": "out_features", "bias": "use_bias"},
      ),
      "Flatten": StandardMap(api="flax.nnx.Flatten"),
      # --- Modular Attention Plugin Selection ---
      "MultiheadAttention": StandardMap(api="flax.nnx.MultiHeadAttention", requires_plugin="repack_attn_flax"),
      "Embedding": StandardMap(api="flax.nnx.Embed", args={"embedding_dim": "features"}),
      "Sequential": StandardMap(api="flax.nnx.Sequential"),
      "BatchNorm": StandardMap(api="flax.nnx.BatchNorm", args={"eps": "epsilon"}, requires_plugin="batch_norm_unwrap"),
      "LayerNorm": StandardMap(api="flax.nnx.LayerNorm", args={"normalized_shape": "num_features", "eps": "epsilon"}),
      # Alias for methods that might be called functionally via nnx.*
      "GELU": StandardMap(api="flax.nnx.gelu"),
      "relu": StandardMap(api="flax.nnx.relu"),
      "softmax": StandardMap(api="flax.nnx.softmax"),
      "log_softmax": StandardMap(api="flax.nnx.log_softmax"),
      "Conv2d": StandardMap(api="flax.nnx.Conv"),
      "Dropout": StandardMap(api="flax.nnx.Dropout"),
      "MaxPool2d": StandardMap(api="flax.nnx.max_pool"),
      # Container Mapping
      "Param": StandardMap(api="flax.nnx.Param"),
      "Variable": StandardMap(api="flax.nnx.Variable"),
      "Cache": StandardMap(api="flax.nnx.Cache"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  # --- Converter ---

  def convert(self, data: Any) -> Any:
    return JaxCoreAdapter().convert(data)

  # --- Wiring ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual wiring specific to Flax.
    """
    self._apply_stack_wiring(snapshot)
    mappings = snapshot.setdefault("mappings", {})

    # Aliases
    for key, variant in mappings.items():
      if variant and "api" in variant:
        api = variant["api"]
        if api.startswith("flax.nnx."):
          mappings[key]["api"] = api.replace("flax.nnx.", "nnx.")

    # Plugins
    for op in ["forward", "__call__", "call"]:
      if op not in mappings or "api" not in mappings[op]:
        mappings[op] = {"requires_plugin": "inject_training_flag"}

    # Containers
    mappings["register_buffer"] = {"requires_plugin": "torch_register_buffer_to_nnx"}
    mappings["register_parameter"] = {"requires_plugin": "torch_register_parameter_to_nnx"}
    mappings["state_dict"] = {"requires_plugin": "torch_state_dict_to_nnx"}
    mappings["load_state_dict"] = {"requires_plugin": "torch_load_state_dict_to_nnx"}
    mappings["parameters"] = {"requires_plugin": "torch_parameters_to_nnx"}

  # --- Examples ---

  @classmethod
  def get_example_code(cls) -> str:
    return """from flax import nnx
import jax.numpy as jnp

class Net(nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        self.linear = nnx.Linear(10, 10, rngs=rngs) 

    def __call__(self, x): 
        x = self.linear(x) 
        return nnx.relu(x) 
"""

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "# Flax NNX State Management\n# See repo for details on nnx.Variable interactions.",
    }
