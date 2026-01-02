"""
Flax NNX Framework Adapter (Level 2).

Extends the JAX core adapter with Flax's Neural Network Extensions (nnx).

- Uses dynamic or snapshot mode discovery.
- Provides clear import alias for `from flax import nnx`.
- Defines the correct base class `flax.nnx.Module`.
- Wires important plugins and structural traits.
"""

import numpy

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional

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
from ml_switcheroo.frameworks.jax import JaxCoreAdapter

# Safe Import Handling for Ghost Mode (WASM/Docs)
try:
  import jax
except ImportError:
  jax = None

try:
  import flax.nnx

  # Explicitly bind the variable name so it is defined in success case
  flax_nnx = flax.nnx
except ImportError:
  flax_nnx = None


@register_framework("flax_nnx")
class FlaxNNXAdapter(JAXStackMixin):
  """
  Adapter class for Flax NNX.

  Inherits from JAXStackMixin for core math/optax behavior and adds:
  - Explicit neural network layers and activations.
  - Correct import aliasing for `from flax import nnx`.
  - Structural traits targeting Flax's nnx Module base.
  """

  display_name: str = "Flax NNX"
  inherits_from: str = "jax"
  ui_priority: int = 15

  def __init__(self) -> None:
    """
    Initializes the adapter.

    - Chooses LIVE mode if `flax.nnx` can be imported.
    - Otherwise, falls back to GHOST mode and loads an API snapshot.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

    if flax_nnx is not None:
      self._flax_available = True
    else:
      self._flax_available = False
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("flax_nnx")
      if not self._snapshot_data:
        logging.warning("Flax NNX not installed and no snapshot found.")

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collect API definitions for discovery.

    Args:
        category (StandardCategory): The API category (layer, activation, etc.)

    Returns:
        List[GhostRef]: List of found API references.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    results: List[GhostRef] = []
    core = JaxCoreAdapter()

    # Use core JAX scanning for losses, optimizers, activations
    if category in [
      StandardCategory.LOSS,
      StandardCategory.OPTIMIZER,
      StandardCategory.ACTIVATION,
    ]:
      results.extend(core.collect_api(category))

    # Add Flax-specific neural layers
    if category == StandardCategory.LAYER:
      results.extend(self._scan_nnx_layers())

    return results

  def _scan_nnx_layers(self) -> List[GhostRef]:
    """
    Scan `flax.nnx` module for classes inheriting from `nnx.Module`, excluding the base Module class.

    Returns:
        List[GhostRef]: Found references.
    """
    if not self._flax_available:
      return []

    found: List[GhostRef] = []
    try:
      from flax import nnx
      import inspect

      for name, obj in inspect.getmembers(nnx):
        if name.startswith("_"):
          continue
        try:
          if inspect.isclass(obj) and issubclass(obj, nnx.Module) and name != "Module":
            found.append(GhostInspector.inspect(obj, f"flax.nnx.{name}"))
        except Exception:
          continue
    except Exception as e:
      logging.debug(f"Error scanning flax.nnx: {e}")

    return found

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """
    Hydrate API ghosts from snapshot data.

    Args:
        category (StandardCategory): Category to filter.

    Returns:
        List[GhostRef]: Hydrated ghost references.
    """
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  @property
  def search_modules(self) -> List[str]:
    """
    Modules to scan for discovery.

    Returns:
        List[str]: Ordered list of module names.
    """
    if self._mode == InitMode.GHOST:
      return []
    return ["jax.numpy", "flax.nnx", "optax"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Returns the base package and alias to guide import injection.
    Used by ImportFixer to map `flax.nnx` root usage to `nnx` alias.

    Returns:
        Tuple[str, str]: (root_package, alias)
    """
    return ("flax.nnx", "nnx")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Declares self namespaces with tiers and recommended aliases.

    Returns:
        Dict[str, ImportConfig]: Mapping of package paths to configs.
    """
    return {
      "flax.linen": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "flax.nnx": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nnx"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for heuristic category assignment during scaffolding.

    Returns:
        Dict[str, List[str]]: Mapping of tiers to regex lists.
    """
    return {"neural": [r"\\.nnx\\.", r"\\.linen\\."], "extras": [r"random\\."]}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Test code templates extended from JAX core.

    Returns:
        Dict[str, str]: Test harness code snippets/templates.
    """
    conf = self.jax_test_config.copy()
    conf["import"] = conf["import"] + "\nimport flax.nnx as nnx"
    return conf

  # --- Verification Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """Imports for Harness generation."""
    return ["from flax import nnx"]

  def get_harness_init_code(self) -> str:
    """Logic to create Flax NNX Rngs."""
    return textwrap.dedent(
      """ 
        def _make_flax_rngs(seed): 
            "Attempts to create a Flax NNX Rngs object." 
            try: 
                return nnx.Rngs(seed) 
            except (ImportError, AttributeError): 
                return "mock_flax_rngs" 
    """
    ).strip()

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Semantic tiers supported by this adapter.

    Returns:
        List[SemanticTier]: Supported tiers.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    return ["rngs"]

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Structural rewriting traits guiding the pivot rewriter.
    Explicitly defines `flax.nnx.Module` to ensure clean inheritance rewriting
    without internal submodule leakage.

    Returns:
        StructuralTraits: Configuration of base class, methods, and injections.
    """
    return StructuralTraits(
      module_base="flax.nnx.Module",
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
    Plugin capabilities indicating required behaviors in the target framework.

    Returns:
        PluginTraits: Flags controlling plugin execution.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=True,
      requires_functional_state=True,
      requires_functional_control_flow=True,
      enforce_purity_analysis=True,
      strict_materialization_method="block_until_ready",
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static standard operation definitions specific to Flax NNX.

    Returns:
        Dict[str, StandardMap]: Mapping of standard op names to framework implementations.
    """
    return {
      "Linear": StandardMap(
        api="flax.nnx.Linear",
        args={"in_features": "in_features", "out_features": "out_features", "bias": "use_bias"},
        # Removed inject_args for rngs: Relying on structural_traits.inject_magic_args for dynamic variable injection
      ),
      "Flatten": StandardMap(api="flax.nnx.Flatten"),
      "MultiheadAttention": StandardMap(api="flax.nnx.MultiHeadAttention", requires_plugin="repack_attn_flax"),
      "Embedding": StandardMap(api="flax.nnx.Embed", args={"embedding_dim": "features"}),
      "Sequential": StandardMap(api="flax.nnx.Sequential"),
      "BatchNorm": StandardMap(api="flax.nnx.BatchNorm", args={"eps": "epsilon"}, requires_plugin="batch_norm_unwrap"),
      "LayerNorm": StandardMap(api="flax.nnx.LayerNorm", args={"normalized_shape": "num_features", "eps": "epsilon"}),
      "GELU": StandardMap(api="flax.nnx.gelu"),
      "relu": StandardMap(api="flax.nnx.relu"),
      "softmax": StandardMap(api="flax.nnx.softmax"),
      "log_softmax": StandardMap(api="flax.nnx.log_softmax"),
      "Conv2d": StandardMap(api="flax.nnx.Conv"),
      "Dropout": StandardMap(api="flax.nnx.Dropout"),
      "MaxPool2d": StandardMap(api="flax.nnx.max_pool"),
      "Param": StandardMap(api="flax.nnx.Param"),
      "Variable": StandardMap(api="flax.nnx.Variable"),
      "Cache": StandardMap(api="flax.nnx.Cache"),
      # --- State Management (Targeting Flax -> Torch) ---
      # This enables converting nnx code back to torch.
      # nnx.BatchStat maps to "Variable" abstract op or dedicated key if defined
      "BatchStat": StandardMap(api="flax.nnx.BatchStat"),
      # --- State Management (Targeting Torch -> Flax) ---
      # Plugins formerly in JaxCore are now correctly placed here in the Neural adapter
      "register_buffer": StandardMap(api="torch_register_buffer_to_nnx", requires_plugin="torch_register_buffer_to_nnx"),
      "register_parameter": StandardMap(
        api="torch_register_parameter_to_nnx", requires_plugin="torch_register_parameter_to_nnx"
      ),
      "state_dict": StandardMap(api="torch_state_dict_to_nnx", requires_plugin="torch_state_dict_to_nnx"),
      "load_state_dict": StandardMap(api="torch_load_state_dict_to_nnx", requires_plugin="torch_load_state_dict_to_nnx"),
      "parameters": StandardMap(api="torch_parameters_to_nnx", requires_plugin="torch_parameters_to_nnx"),
      "SiLU": StandardMap(api="flax.nnx.silu"),
      "ModuleList": StandardMap(api="flax.nnx.List"),
      "TensorType": StandardMap(api="jax.Array"),
      "Arange": StandardMap(api="jax.numpy.arange"),
      "Ones": StandardMap(api="jax.numpy.ones"),
      "Pad": StandardMap(
        api="jax.numpy.pad",
        args={"input": "array", "pad": "pad_width", "value": "constant_values"},
        requires_plugin="padding_converter",
      ),
      "AssertClose": StandardMap(
        api="numpy.testing.assert_allclose", args={"expected": "desired"}, required_imports=["import numpy"]
      ),
    }

  def convert(self, data: Any) -> Any:
    """
    Converts generic data to framework-specific Pytree/arrays.

    Returns:
        Converted data tailored to JAX/Flax ecosystem.
    """
    return JaxCoreAdapter().convert(data)

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual wiring and modifies the snapshot to alias 'flax.nnx.' to 'nnx.'.

    Adds plugin wiring for key interface methods.

    Args:
        snapshot (Dict[str, Any]): The mapping snapshot dictionary.
    """
    self._apply_stack_wiring(snapshot)
    mappings = snapshot.setdefault("mappings", {})

    # Replace long module prefix to use alias 'nnx.'
    for key, variant in mappings.items():
      if variant and "api" in variant:
        api = variant["api"]
        if api.startswith("flax.nnx."):
          mappings[key]["api"] = api.replace("flax.nnx.", "nnx.")

    # Ensure forwarding methods have proper plugin wiring for training flag injection
    for op in ["forward", "__call__", "call"]:
      if op not in mappings or "api" not in mappings[op]:
        mappings[op] = {"requires_plugin": "inject_training_flag"}

    # Note: Plugins for container state (register_buffer, etc.) are now statically defined in definitions
    # so we don't need to manually wire them here if definitions are merged.
    # We keep the fallback wiring just in case Ghost Mode lacks static defs.
    mappings.setdefault("register_buffer", {"requires_plugin": "torch_register_buffer_to_nnx"})
    mappings.setdefault("register_parameter", {"requires_plugin": "torch_register_parameter_to_nnx"})
    mappings.setdefault("state_dict", {"requires_plugin": "torch_state_dict_to_nnx"})
    mappings.setdefault("load_state_dict", {"requires_plugin": "torch_load_state_dict_to_nnx"})
    mappings.setdefault("parameters", {"requires_plugin": "torch_parameters_to_nnx"})

  @classmethod
  def get_example_code(cls) -> str:
    """
    Example neural network showcasing Flax NNX idiomatic usage.

    Returns:
        str: Example source code string.
    """
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
    """
    Provides tier-specific example usages for documentation and tests.

    Returns:
        Dict[str, str]: Dictionary mapping tier names to code snippets.
    """
    return {
      "tier2_neural": self.get_example_code(),
      "tier3_extras": "# Flax NNX State Management\n# See repo for details on nnx.Variable interactions.",
    }

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns the official Flax documentation URL for a given API string.
    Defaults to ReadTheDocs search query for robustness with new NNX APIs.

    Args:
        api_name (str): The fully qualified API name.

    Returns:
        Optional[str]: The URL to the documentation page.
    """
    # Flax NNX APIs change frequently, search is safest
    return f"https://flax.readthedocs.io/en/latest/search.html?q={api_name}"
