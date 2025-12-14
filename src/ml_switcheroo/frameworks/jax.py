"""
JAX Framework Adapter.

Integrates:
- JAX/Flax/Optax Ecosystem.
- Hybrid Discovery (Live + Ghost).
"""

import inspect
import sys
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
  import jax
  import jax.numpy as jnp
except ImportError:
  jax = None
  jnp = None

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
from ml_switcheroo.frameworks.optax_shim import OptaxScanner
from ml_switcheroo.frameworks.flax_shim import FlaxScanner


@register_framework("jax")
class JaxAdapter:
  """Adapter for JAX / Flax / Optax."""

  display_name: str = "JAX / Flax"
  inherits_from: Optional[str] = None
  ui_priority: int = 10

  def __init__(self):
    """
    Initialize JAX Adapter.
    Checks for JAX presence. If missing, enters Ghost Mode.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if jax is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("jax")
      if not self._snapshot_data:
        logging.warning("JAX not installed and no snapshot found. Scanning unavailable.")

    @classmethod
    def get_example_code(cls) -> str:
      return cls().get_tiered_examples()["tier1_math"]

    def get_tiered_examples(self) -> Dict[str, str]:
      return {
        "tier1_math": """import jax.numpy as jnp

def math_ops(x, y):
  # Tier 1: Array API Standard
  a = jnp.abs(x)
  b = jnp.add(a, y)
  return jnp.mean(b)""",
        "tier2_neural": """from flax import nnx
import jax.numpy as jnp

class Net(nnx.Module):
  # Tier 2: Neural State & Layers
  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(10, 10, rngs=rngs)

  def __call__(self, x):
    x = self.linear(x)
    return nnx.relu(x)""",
        "tier3_extras": """import jax
from jax import random

def stochastic_step(x, seed):
  # Tier 3: Extras (Random State)
  rng = random.PRNGKey(seed)
  rng, key = random.split(rng)

  # Explicit RNG passing
  return random.normal(key, x.shape)""",
      }

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
    """
    Scans the fragmented JAX ecosystem.
    - LOSS/OPTIMIZER -> Optax
    - LAYER -> Flax Linen
    - ACTIVATION -> Flax Linen or JAX NN
    """
    results = []

    if category == StandardCategory.LOSS:
      results.extend(OptaxScanner.scan_losses())

    elif category == StandardCategory.OPTIMIZER:
      results.extend(OptaxScanner.scan_optimizers())

    elif category == StandardCategory.LAYER:
      results.extend(FlaxScanner.scan_layers())

    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_jax_activations())

    return results

  def _scan_jax_activations(self) -> List[GhostRef]:
    """Internal scanner for built-in JAX activations."""
    # Access global jax check
    if jax is None:
      return []
    found = []
    # jax.nn contains relu, silu, etc.
    try:
      # Local import 'as' to prevent shadowing global 'jax' variable in this scope
      import jax.nn as jax_nn

      for name, obj in inspect.getmembers(jax_nn):
        if name.startswith("_"):
          continue

        if inspect.isfunction(obj):
          # Filter common activation names
          if name in ["relu", "gelu", "silu", "elu", "sigmoid", "tanh", "softmax", "log_softmax"]:
            ref = GhostInspector.inspect(obj, f"jax.nn.{name}")
            found.append(ref)
    except ImportError:
      pass
    return found

  # --- Metadata ---

  @property
  def search_modules(self) -> List[str]:
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "flax.nnx", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("jax.numpy", "jnp")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.linen", r"Module$"], "extras": [r"random\.", r"pmap", r"vmap", r"jit"]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="nnx.Module",  # UPDATED: Use alias nnx.Module
      forward_method="__call__",
      inject_magic_args=[("rngs", "nnx.Rngs")],  # UPDATED: Use alias nnx.Rngs
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

  # --- Syntax Generation ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

  def get_serialization_imports(self) -> List[str]:
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"orbax.checkpoint.PyTreeCheckpointer().save(directory={file_arg}, item={object_arg})"
    elif op == "load":
      return f"orbax.checkpoint.PyTreeCheckpointer().restore({file_arg})"
    return ""

  def convert(self, data):
    try:
      import jax.numpy as jnp
    except ImportError:
      return data
    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return jnp.array(data)
    return data

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Injects JAX/Flax specific plugin hooks and templates."""
    mappings = snapshot.setdefault("mappings", {})
    templates = snapshot.setdefault("templates", {})
    # UPDATED: We use the imports config to drive aliases
    imports = snapshot.setdefault("imports", {})

    # --- 1. Enforce Aliases (Paths found by Sync are rewritten here) ---

    # A. Rewrite Discovery Paths (jax.numpy -> jnp, flax.nnx -> nnx)
    for key, variant in mappings.items():
      if variant and "api" in variant:
        api = variant["api"]
        if api.startswith("jax.numpy."):
          mappings[key]["api"] = api.replace("jax.numpy.", "jnp.")
        elif api.startswith("flax.nnx."):
          mappings[key]["api"] = api.replace("flax.nnx.", "nnx.")

    # B. Configure Import Injection
    # When code uses "nnx", ImportFixer will inject "from flax import nnx"
    imports["flax.nnx"] = {"root": "flax", "sub": "nnx", "alias": "nnx"}

    # Fix: Ensure logic maps to 'jnp.abs' regardless of casing in specs
    # This prevents 'torch.abs' falling through if specs use 'abs' vs 'Abs'
    mappings["Abs"] = {"api": "jnp.abs"}
    mappings["abs"] = {"api": "jnp.abs"}

    # Removal: Do NOT inject torch.nn -> flax.nnx.
    # This allows ImportFixer to prune torch.nn when it is unused.

    # --- 2. Plugin Configuration ---

    # Argument Packing (permute_dims -> transpose with tuple)
    mappings["permute_dims"] = {"api": "jnp.transpose", "requires_plugin": "pack_varargs"}

    # Einsum Normalization
    if "Einsum" in mappings:
      mappings["Einsum"]["requires_plugin"] = "einsum_normalizer"
      if "api" not in mappings["Einsum"] or mappings["Einsum"]["api"] == "jax.numpy.einsum":
        mappings["Einsum"]["api"] = "jnp.einsum"

    # Method -> Property Swaps (e.g. .size() -> .shape)
    mappings["size"] = {"api": "shape", "requires_plugin": "method_to_property"}
    mappings["data_ptr"] = {"api": "data", "requires_plugin": "method_to_property"}

    # State Flag Injection (Training/Eval modes)
    for op in ["forward", "__call__", "call"]:
      if op not in mappings or "api" not in mappings[op]:
        mappings[op] = {"requires_plugin": "inject_training_flag"}

    # State Container Plugins (Torch -> Flax NNX mapping)
    mappings["register_buffer"] = {"requires_plugin": "torch_register_buffer_to_nnx"}
    mappings["register_parameter"] = {"requires_plugin": "torch_register_parameter_to_nnx"}
    mappings["state_dict"] = {"requires_plugin": "torch_state_dict_to_nnx"}
    mappings["load_state_dict"] = {"requires_plugin": "torch_load_state_dict_to_nnx"}
    mappings["parameters"] = {"requires_plugin": "torch_parameters_to_nnx"}

    # Optimizer Wiring (Torch -> Optax Translation)
    mappings["step"] = {"requires_plugin": "optimizer_step"}
    mappings["zero_grad"] = {"requires_plugin": "optimizer_zero_grad"}

    # Wire common Optimizers to the constructor stripping hook
    for opt_name in ["Adam", "SGD", "RMSprop", "AdamW", "Adagrad", "LBFGS"]:
      if opt_name not in mappings:
        mappings[opt_name] = {}
      mappings[opt_name]["requires_plugin"] = "optimizer_constructor"
      if "api" not in mappings[opt_name]:
        mappings[opt_name]["api"] = f"optax.{opt_name.lower()}"

    # Loop Templates (Hints for Escape Hatches or Plugins)
    templates["fori_loop"] = "val = jax.lax.fori_loop({start}, {stop}, lambda i, val: {body}, {init_val})"
    templates["scan"] = "carry, stacked = jax.lax.scan(lambda c, x: {body}, {init}, {xs})"
