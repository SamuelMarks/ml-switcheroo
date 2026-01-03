"""
Apple MLX Framework Adapter.

This module provides the adapter for Apple's MLX array framework.
It supports:
1.  **Unified Memory math**: Mapping ``mlx.core`` operations.
2.  **Neural Networks**: Mapping ``mlx.nn`` layers and containers.
3.  **Discovery**: Runtime introspection of the MLX API surface.
4.  **Types**: Mapping Abstract Types to ``mlx.core`` dtypes (e.g. ``mx.float32``).
5.  **Casting**: Generic casting plugin integration via ``.astype()``.

Definitions are loaded from `frameworks/definitions/mlx.json`.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np

from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
  InitMode,
)
from ml_switcheroo.frameworks.loader import load_definitions

# Conditional import to allow loading in environments without MLX
try:
  import mlx.core
  import mlx.nn
  import mlx.optimizers
  import mlx.utils
except ImportError:
  pass


@register_framework("mlx")
class MLXAdapter:
  """
  Adapter for Apple MLX (Silicon-optimized tensor framework).
  """

  display_name: str = "Apple MLX"
  inherits_from: Optional[str] = None
  ui_priority: int = 50

  @property
  def search_modules(self) -> List[str]:
    """
    Returns list of MLX submodules to scan during discovery.

    Returns:
        List[str]: Module list.
    """
    return [
      "mlx.core",
      "mlx.nn",
      "mlx.optimizers",
      "mlx.core.fft",
      "mlx.core.linalg",
      "mlx.core.random",
    ]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Submodules safe to avoid during recursion.

    Returns:
        Set[str]: Default empty set.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Default alias for core array operations: ``import mlx.core as mx``.

    Returns:
        Tuple[str, str]: ("mlx.core", "mx").
    """
    return ("mlx.core", "mx")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Self-declaration of namespaces.

    Returns:
        Dict[str, ImportConfig]: Namespace map.
    """
    return {
      "mlx.core": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="mx"),
      "mlx.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "mlx.optimizers": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optim"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for categorizing discovered APIs into Tiers.

    Returns:
        Dict[str, List[str]]: Heuristics map.
    """
    return {
      "neural": [r"\\.nn\\."],
      "extras": [r"random\\."],
      "optimizer": [r"\\.optimizers\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Templates for generating physical verification tests.

    Returns:
        Dict[str, str]: Templates.
    """
    return {
      "import": "import mlx.core as mx\nimport numpy as np",
      "convert_input": "mx.array({np_var})",
      "to_numpy": "np.array({res_var})",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Imports for harness.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Initialization code.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns code to convert MLX arrays (which have .tolist()) to NumPy.

    Returns:
        str: Python logic for conversion.
    """
    return "if hasattr(obj, 'tolist'): return np.array(obj.tolist())"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns supported semantic tiers (Array, Neural, Extras).

    Returns:
        List[SemanticTier]: Supported Tiers.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    Implicit RNG arguments.

    Returns:
        List[str]: Empty.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines structural rewriting rules (Classes, Methods, Init).

    Updated to strip 'rngs' argument coming from Flax NNX, as MLX
    handles initialization statefully/eagerly.

    Returns:
        StructuralTraits: Config object.
    """
    return StructuralTraits(
      module_base="mlx.nn.Module",
      forward_method="__call__",
      requires_super_init=True,
      auto_strip_magic_args=True,  # Decoupled
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Plugin behavior configuration.

    Returns:
        PluginTraits: Config object.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,  # MLX uses implicit RNG in mlx.core.random usually
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for MLX mappings.
    Loaded dynamically from `frameworks/definitions/mlx.json`.

    Returns:
        Dict[str, StandardMap]: Definitions map.
    """
    return load_definitions("mlx")

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns list of global seed setters.

    Returns:
        List[str]: Method names.
    """
    return ["seed", "random.seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Performs runtime introspection to discover available MLX APIs.

    Args:
        category (StandardCategory): Category to scan.

    Returns:
        List[GhostRef]: Found items.
    """
    results = []
    try:
      import mlx.core
      import mlx.nn
      import mlx.optimizers
      import inspect

      if category == StandardCategory.LAYER:
        for name, obj in inspect.getmembers(mlx.nn):
          if not name.startswith("_") and inspect.isclass(obj) and name[0].isupper():
            results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

      if category == StandardCategory.ACTIVATION:
        target_names = {
          "relu",
          "gelu",
          "silu",
          "sigmoid",
          "tanh",
          "softmax",
          "elu",
        }
        for name, obj in inspect.getmembers(mlx.nn):
          if name.lower() in target_names:
            results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

      if category == StandardCategory.LOSS:
        if hasattr(mlx.nn, "losses"):
          for name, obj in inspect.getmembers(mlx.nn.losses):
            if inspect.isfunction(obj) or inspect.isclass(obj):
              if "loss" in name.lower():
                results.append(GhostInspector.inspect(obj, f"mlx.nn.losses.{name}"))

      if category == StandardCategory.OPTIMIZER:
        for name, obj in inspect.getmembers(mlx.optimizers):
          if inspect.isclass(obj) and not name.startswith("_"):
            results.append(GhostInspector.inspect(obj, f"mlx.optimizers.{name}"))

    except ImportError as e:
      logging.debug(f"Could not inspect MLX: {e}")
      pass

    return results

  def convert(self, data: Any) -> Any:
    """
    Converts input data (NumPy/List) to MLX Tensor for verification.

    Args:
        data (Any): Input.

    Returns:
        Any: MLX Array or original.
    """
    try:
      import mlx.core as mx

      if isinstance(data, (np.ndarray, list, tuple, np.generic)):
        return mx.array(data)
    except ImportError:
      pass
    return data

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns default example.

    Returns:
        str: Code.
    """
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns MLX idiomatic examples used for validity testing.

    Returns:
        Dict[str, str]: Example maps.
    """
    return {
      "tier1_math": """import mlx.core as mx

def math_ops(x, y):
    # Tier 1: Unified Buffer Architecture Math
    # MLX uses lazy evaluation by default
    a = mx.abs(x)
    b = mx.add(a, y)

    # Reductions
    return mx.mean(b, axis=0)
""",
      "tier2_neural": """import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class MLP(nn.Module):
    # Tier 2: Neural Modules
    # Inherits from nn.Module, uses __call__ for inference
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.layers = [
            nn.Linear(in_dims, 64),
            nn.ReLU(),
            nn.Linear(64, out_dims)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
        
def train_step(model, optimizer, x, y):
    # Backward pass handled by value_and_grad via mx.compile typically
    pass
""",
      "tier3_extras": """import mlx.core as mx

def compute_on_gpu(x):
    # Tier 3: Extras (Streams & Devices)
    # Explicitly schedule computation on the GPU stream
    with mx.stream(mx.gpu):
        y = mx.array(x) * 2

        # Trigger evaluation (sync)
        mx.eval(y)
        return y
""",
    }

  # --- Syntax Generators ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns device constructor syntax.

    Args:
        device_type: Device description.
        device_index: Device index.

    Returns:
        str: Generated code.
    """
    clean_type = device_type.strip("'\"").lower()
    if clean_type in ("cuda", "gpu", "mps"):
      backend = "mx.gpu"
    else:
      backend = "mx.cpu"
    args = [backend]
    if device_index:
      args.append(str(device_index))
    return f"mx.Device({', '.join(args)})"

  def get_device_check_syntax(self) -> str:
    """
    Check if default device is GPU.
    Note: MLX Unified Memory doesn't have strict 'is_available' but we check backend.

    Returns:
        str: Code string.
    """
    return "mx.default_device() == mx.gpu"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    MLX usually uses implicit state, but if explicit mode is requested,
    return 'pass' as split logic differs significantly.

    Returns:
        str: "pass".
    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """
    Returns imports for serialization.

    Returns:
        List[str]: Imports.
    """
    return ["import mlx.core as mx"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns save/load syntax.

    Args:
        op: 'save' or 'load'.
        file_arg: Target file path.
        object_arg: Object name.

    Returns:
        str: Code string.
    """
    if op == "save" and object_arg:
      return f"mx.save({file_arg}, {object_arg})"
    elif op == "load":
      return f"mx.load({file_arg})"
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual wiring for MLX.
    Overrides/Patches snapshot items that cannot be statically defined.

    Args:
        snapshot: Snapshotdict.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Generates documentation URL for MLX APIs using autosummary pattern.

    Args:
        api_name: Fully qualified API string.

    Returns:
        Optional[str]: URL.
    """
    return f"https://ml-explore.github.io/mlx/build/html/python/_autosummary/{api_name}.html"
