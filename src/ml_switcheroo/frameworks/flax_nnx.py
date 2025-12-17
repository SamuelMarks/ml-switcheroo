"""
Flax NNX Framework Adapter (Level 2).

This adapter builds upon the core JAX stack to support the **Flax NNX**
neural network library. It inherits math/optimizer logic from `JAXStackMixin`
but implements unique structural traits for NNX Modules.

Key Features:
- Inheritance from `jax` core via Mixin reuse.
- Rewriting `torch.nn.Module` -> `flax.nnx.Module`.
- Handling `rngs` state threading for stochastic layers.
- Maps core Neural Ops to ensure successful transpilation if not pre-seeded.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

try:
  import flax.nnx
except ImportError:
  flax_nnx = None

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
from ml_switcheroo.frameworks.flax_shim import FlaxScanner
from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin

# Import core adapter for shared discovery logic re-use if needed
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


@register_framework("flax_nnx")
class FlaxNNXAdapter(JAXStackMixin):
  """
  Adapter for the Flax NNX Framework.
  """

  display_name: str = "Flax NNX"

  # Inherit mappings from 'jax' (the Core adapter)
  inherits_from: str = "jax"

  ui_priority: int = 15

  def __init__(self):
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    # Check for Metadata snapshot availability
    # We try to load 'flax_nnx' snapshot, falling back to ghost logic
    try:
      import flax.nnx
    except ImportError:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("flax_nnx")
      if not self._snapshot_data:
        logging.warning("Flax NNX not installed and no snapshot found.")

  # --- Discovery ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API definitions.
    Leverages JaxCore logic for shared categories (Loss/Activations) via composition
    if living in the same environment, or scans Flax-specific layers.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    # Live Scanning
    results = []

    # 1. Use Core JAX capabilities (Optimization/Loss/Activations)
    # We instantiate a temporary core adapter to reuse its scanning logic
    # This prevents code duplication.
    core = JaxCoreAdapter()
    if category in [StandardCategory.LOSS, StandardCategory.OPTIMIZER, StandardCategory.ACTIVATION]:
      results.extend(core._collect_live(category))

    # 2. Add Flax Specifics (Layers)
    if category == StandardCategory.LAYER:
      results.extend(FlaxScanner.scan_layers())

    return results

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    # Includes Core JAX modules plus Flax NNX specifics
    return ["jax.numpy", "flax.nnx", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    # Default alias for the *primary* interaction point of this framework (Neural)
    # Tuple returned: (module_path, alias)
    return ("flax.nnx", "nnx")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.nnx\.", r"\.linen\."], "extras": [r"random\."]}

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines Flax NNX specific structural transformations (Level 2).
    """
    return StructuralTraits(
      module_base="nnx.Module",
      forward_method="__call__",
      inject_magic_args=[("rngs", "nnx.Rngs")],
      requires_super_init=False,
      # JIT static args inherited concept
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "dtype"],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Static definitions to ensure core ops work without discovery."""
    return {
      "Linear": StandardMap(
        api="flax.nnx.Linear", args={"in_features": "in_features", "out_features": "out_features", "bias": "use_bias"}
      ),
      "relu": StandardMap(api="flax.nnx.relu"),
      "gelu": StandardMap(api="flax.nnx.gelu"),
      "softmax": StandardMap(api="flax.nnx.softmax"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  # --- Converter ---

  def convert(self, data):
    # Delegate to JAX core conversion
    return JaxCoreAdapter().convert(data)

  # --- Wiring ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies Stack wiring + Flax NNX specific logic.
    """
    # 1. Apply Stack Logic (L0 & L1) from Mixin
    self._apply_stack_wiring(snapshot)

    mappings = snapshot.setdefault("mappings", {})
    imports = snapshot.setdefault("imports", {})

    # 2. Configure Aliases
    # Override discovered API paths to use standard aliases
    for key, variant in mappings.items():
      if variant and "api" in variant:
        api = variant["api"]
        # Convert explicit flax.nnx.Linear -> nnx.Linear
        if api.startswith("flax.nnx."):
          mappings[key]["api"] = api.replace("flax.nnx.", "nnx.")

    # Ensure 'nnx' usage triggers 'from flax import nnx'
    imports["flax.nnx"] = {"root": "flax", "sub": "nnx", "alias": "nnx"}

    # 3. Configure State Plugins
    # Wire standard neural methods to state injection hooks
    for op in ["forward", "__call__", "call"]:
      if op not in mappings or "api" not in mappings[op]:
        mappings[op] = {"requires_plugin": "inject_training_flag"}

    # Wire PyTorch -> NNX containers
    mappings["register_buffer"] = {"requires_plugin": "torch_register_buffer_to_nnx"}
    mappings["register_parameter"] = {"requires_plugin": "torch_register_parameter_to_nnx"}
    mappings["state_dict"] = {"requires_plugin": "torch_state_dict_to_nnx"}
    mappings["load_state_dict"] = {"requires_plugin": "torch_load_state_dict_to_nnx"}
    mappings["parameters"] = {"requires_plugin": "torch_parameters_to_nnx"}

    # 4. Neural Layer Mappings (Safety Injection)
    # Ensures Rewriter can map 'Linear'/'relu' even if snapshot/discovery failed.
    # Mappings are for when flax_nnx is the TARGET.
    # NOTE: When flax_nnx is the SOURCE, these mappings are loaded into Reverse Index by Manager
    # allowing detection of 'flax.nnx.Linear'.

    # Linear: Use nnx.Linear alias
    if "Linear" not in mappings:
      mappings["Linear"] = {
        "api": "flax.nnx.Linear",
        "args": {"in_features": "in_features", "out_features": "out_features", "bias": "use_bias"},
      }

    # Activations
    # flax.nnx often re-exports jax.nn or has its own wrappers.
    if "relu" not in mappings:
      mappings["relu"] = {"api": "flax.nnx.relu"}
    if "gelu" not in mappings:
      mappings["gelu"] = {"api": "flax.nnx.gelu"}
    if "softmax" not in mappings:
      mappings["softmax"] = {"api": "flax.nnx.softmax"}

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
