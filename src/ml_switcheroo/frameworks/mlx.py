"""
Apple MLX Framework Adapter.

This module provides the adapter for Apple's MLX array framework.
It supports:
1.  **Unified Memory math**: Mapping `mlx.core` operations.
2.  **Neural Networks**: Mapping `mlx.nn` layers and containers.
3.  **Discovery**: Runtime introspection of the MLX API surface.
4.  **Types**: Mapping Abstract Types to `mlx.core` dtypes (e.g. `mx.float32`).
5.  **Casting**: Generic casting plugin integration via `.astype()`.

Refactor: Distributed definitions populated for MLX specific Layers, Ops, Optimizers, and Types.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging

import numpy as np

from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import register_framework, StructuralTraits, StandardCategory, StandardMap

# Conditional import to allow loading in environments without MLX
try:
  import mlx.core
  import mlx.nn
  import mlx.optimizers
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
    """
    return ["mlx.core", "mlx.nn", "mlx.optimizers", "mlx.core.fft", "mlx.core.linalg", "mlx.core.random"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Default alias for core array operations: `import mlx.core as mx`.
    """
    return ("mlx.core", "mx")

  @property
  def import_namespaces(self) -> Dict[str, Dict[str, str]]:
    """
    Defines namespace mapping for source-to-target imports.
    Maps `torch.nn` -> `mlx.nn` (aliased as `nn`) and `mlx.core` mappings.
    """
    return {
      "torch.nn": {"root": "mlx", "sub": "nn", "alias": "nn"},
      "flax.nnx": {"root": "mlx", "sub": "nn", "alias": "nn"},
      "mlx.core": {"root": "mlx", "sub": "core", "alias": "mx"},
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for categorizing discovered APIs into Tiers.
    """
    return {"neural": [r"\\.nn\\."], "extras": [r"random\\."], "optimizer": [r"\\.optimizers\\."]}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Templates for generating physical verification tests.
    """
    return {
      "import": "import mlx.core as mx\nimport numpy as np",
      "convert_input": "mx.array({np_var})",
      "to_numpy": "np.array({res_var})",
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns supported semantic tiers (Array, Neural, Extras).
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines structural rewriting rules (Classes, Methods, Init).

    Updated to strip 'rngs' argument coming from Flax NNX, as MLX
    handles initialization statefully/eagerly.
    """
    return StructuralTraits(
      module_base="mlx.nn.Module", forward_method="__call__", requires_super_init=True, strip_magic_args=["rngs"]
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for MLX mappings.
    Covers Optimization, Math, Layers, Compilation, Types, and Casting.
    """
    return {
      "Abs": StandardMap(api="mx.abs"),
      "Mean": StandardMap(api="mx.mean"),
      "Linear": StandardMap(api="mlx.nn.Linear", args={"in_features": "input_dims", "out_features": "output_dims"}),
      "permute_dims": StandardMap(api="mx.transpose", pack_to_tuple="axes"),
      # --- Optimization ---
      "Adam": StandardMap(api="mlx.optimizers.Adam", args={"lr": "learning_rate"}, requires_plugin="mlx_optimizer_init"),
      "SGD": StandardMap(api="mlx.optimizers.SGD", args={"lr": "learning_rate"}, requires_plugin="mlx_optimizer_init"),
      "step": StandardMap(api="optimizer_step", requires_plugin="mlx_optimizer_step"),
      "zero_grad": StandardMap(api="optimizer_zero_grad", requires_plugin="mlx_zero_grad"),
      # --- Arrays ---
      "randn": StandardMap(api="mlx.random.normal"),
      "ArgMax": StandardMap(api="mlx.core.argmax", args={"dim": "axis", "keepdim": "keepdims"}),
      "ArgMin": StandardMap(api="mlx.core.argmin", args={"dim": "axis", "keepdim": "keepdims"}),
      # --- Layers ---
      "Embedding": StandardMap(api="mlx.nn.Embedding", args={"embedding_dim": "dims"}),
      "BatchNorm": StandardMap(api="mlx.nn.BatchNorm"),
      "LayerNorm": StandardMap(api="mlx.nn.LayerNorm", args={"normalized_shape": "dims"}),
      "GELU": StandardMap(api="mlx.nn.GELU"),
      "OneHot": StandardMap(api="mlx.nn.OneHot"),
      "relu": StandardMap(api="mlx.nn.relu"),
      "softmax": StandardMap(api="mlx.core.softmax"),
      "log_softmax": StandardMap(api="mlx.nn.log_softmax"),
      # --- Extras including compilation and sync ---
      "Compile": StandardMap(api="mlx.core.compile", requires_plugin="mlx_compiler"),
      "Synchronize": StandardMap(api="mx.eval", requires_plugin="mlx_synchronize"),
      "no_grad": StandardMap(api="contextlib.nullcontext", requires_plugin="context_to_function_wrap"),
      "enable_grad": StandardMap(api="contextlib.nullcontext", requires_plugin="context_to_function_wrap"),
      "Variable": StandardMap(api="mlx.core.array"),
      # --- Types ---
      "Float32": StandardMap(api="mx.float32"),
      "Float16": StandardMap(api="mx.float16"),
      # Mapping Float64 to Float32 as Metal/MLX generally optimizes for lower precision,
      # and missing explicit Double support in some contexts triggers fallback.
      "Float64": StandardMap(api="mx.float32"),
      "Int64": StandardMap(api="mx.int64"),
      "Int32": StandardMap(api="mx.int32"),
      "Int16": StandardMap(api="mx.int16"),
      "UInt8": StandardMap(api="mx.uint8"),
      "Bool": StandardMap(api="mx.bool_"),
      # --- Casting (via Plugin) ---
      "CastFloat": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastDouble": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastHalf": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastLong": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastInt": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastShort": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastByte": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastBool": StandardMap(api="astype", requires_plugin="type_methods"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    """Returns list of global seed setters."""
    return ["seed", "random.seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Performs runtime introspection to discover available MLX APIs.
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
        target_names = {"relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "elu"}
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
    """
    try:
      import mlx.core as mx
      # import numpy as np # already imported globally

      if isinstance(data, (np.ndarray, list, tuple, np.generic)):
        return mx.array(data)
    except ImportError:
      pass
    return data

  @classmethod
  def get_example_code(cls) -> str:
    """Returns default example."""
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns MLX idiomatic examples used for validity testing.
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

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Returns device constructor syntax."""
    clean_type = device_type.strip("'\"").lower()
    if clean_type in ("cuda", "gpu", "mps"):
      backend = "mx.gpu"
    else:
      backend = "mx.cpu"
    args = [backend]
    if device_index:
      args.append(str(device_index))
    return f"mx.Device({', '.join(args)})"

  def get_serialization_imports(self) -> List[str]:
    """Returns imports for serialization."""
    return ["import mlx.core as mx"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Returns save/load syntax."""
    if op == "save" and object_arg:
      return f"mx.save({file_arg}, {object_arg})"
    elif op == "load":
      return f"mx.load({file_arg})"
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies manual wiring for MLX.
    Overrides/Patches snapshot items that cannot be statically defined.
    """
    # Wiring is primarily handled by definitions property now.
    # Only dynamic adjustments or complex conditional logic would go here.
    pass
