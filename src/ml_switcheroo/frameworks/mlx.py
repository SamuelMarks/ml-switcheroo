"""Apple MLX Framework Adapter.

This module provides the adapter for Apple's MLX array framework.
It supports:
1.  **Unified Memory math**: Mapping ``mlx.core`` operations.
2.  **Neural Networks**: Mapping ``mlx.nn`` layers and containers.
3.  **Discovery**: Runtime introspection of the MLX API surface.
4.  **Types**: Mapping Abstract Types to ``mlx.core`` dtypes (e.g. ``mx.float32``).
5.  **Casting**: Generic casting plugin integration via ``.astype()``.
6.  **Weight Migration**: Loading/saving .npz or .safetensors files (via stubs/core).

Definitions are loaded from `frameworks/definitions/mlx.json`.
"""

from ml_switcheroo.enums import RuntimeTensorOrValue

import logging
from types import ModuleType

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.frameworks.base import ImportConfig, PluginTraits, StandardMap, StructuralTraits, register_framework
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.frameworks.mlx_io import MlxIOMixin

_np_mod: ModuleType | None = None
np: ModuleType | None = None
try:
  import numpy as _np

  _np_mod = _np
  np = _np
except (ImportError, AttributeError, TypeError, ValueError) as exc:
  logging.getLogger(__name__).debug("Failed to import numpy in mlx: %s", exc)
  _np_mod = None
  np = None


@register_framework("mlx")
class MLXAdapter(MlxIOMixin):
  """Adapter for Apple MLX (Silicon-optimized tensor framework)."""

  display_name: str = "Apple MLX"
  inherits_from: str | None = None
  ui_priority: int = 50

  @property
  def import_alias(self) -> tuple[str, str]:
    """Supply alias for core array operations: ``import mlx.core as mx``.

    Returns:
        Tuple[str, str]: ("mlx.core", "mx").

    """
    return "mlx.core", "mx"

  @property
  def import_namespaces(self) -> dict[str, ImportConfig]:
    """Self-declaration of namespaces.

    Returns:
        Dict[str, ImportConfig]: Namespace map.

    """
    return {
      "mlx.core": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="mx"),
      "mlx.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "mlx.optimizers": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optim"),
    }

  @property
  def test_config(self) -> dict[str, str]:
    """Template for generating physical verification tests.

    Returns:
        Dict[str, str]: Templates.

    """
    return {
      "import": "import mlx.core as mx\nimport numpy as np",
      "convert_input": "mx.array({np_var})",
      "to_numpy": "np.array({res_var})",
    }

  @property
  def harness_imports(self) -> list[str]:
    """Import for harness.

    Returns:
        List[str]: Empty list.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Initialize code.

    Returns:
        str: Empty string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Return code to convert MLX arrays (which have .tolist()) to NumPy.

    Returns:
        str: Python logic for conversion.

    """
    return "if hasattr(obj, 'tolist'): return np.array(obj.tolist())"

  @property
  def supported_tiers(self) -> list[SemanticTier]:
    """Return supported semantic tiers (Array, Neural, Extras).

    Returns:
        List[SemanticTier]: Supported Tiers.

    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> list[str]:
    """Implicit RNG arguments.

    Returns:
        List[str]: Empty.

    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Define structural rewriting rules (Classes, Methods, Init).

    Updated to strip 'rngs' argument coming from Flax NNX, as MLX
    handles initialization statefully/eagerly.

    Returns:
        StructuralTraits: Config object.

    """
    return StructuralTraits(
      module_base="mlx.nn.Module",
      tensor_type="mlx.core.array",
      forward_method="__call__",
      requires_super_init=True,
      auto_strip_magic_args=True,
      default_imports=["import mlx.core as mx", "import mlx.nn as nn"],
      layer_namespace_prefix="nn",
      functional_namespace_prefix="mx",
      module_alias_strip=("mlx.nn.", "nn."),
      functional_namespace_skip="mlx.core.",
      sharding_style="distributed_shard",
      rng_seed_function=("mx", "random", "seed"),
      layer_name_rewrites={
        "SwiGLU": "silu",
        "RoPE": "RoPE",
        "VisionPatchEmbedding": "Conv2d",
      },
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Plugin behavior configuration.

    Returns:
        PluginTraits: Config object.

    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> dict[str, StandardMap]:
    """Static definitions for MLX mappings.

    Loaded dynamically from `frameworks/definitions/mlx.json`.

    Returns:
        Dict[str, StandardMap]: Definitions map.

    """
    defs = load_definitions("mlx")
    if "Linear" not in defs:
      defs["Linear"] = StandardMap(
        api="mlx.nn.Linear",
        args={"in_features": "input_dims", "out_features": "output_dims"},
      )
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="mlx.nn.Conv2d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    return defs

  @property
  def rng_seed_methods(self) -> list[str]:
    """Return list of global seed setters.

    Returns:
        List[str]: Method names.

    """
    return ["seed", "random.seed"]

  def convert(self, data: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
    """Convert input data (NumPy/List) to MLX Tensor for verification.

    Args:
        data: Input.

    Returns:
        MLX Array or original.

    """
    try:
      import mlx.core as mx

      if np is not None and isinstance(data, (np.ndarray, np.generic)):
        res_np: RuntimeTensorOrValue = mx.array(data)
        return res_np
      elif isinstance(data, (list, tuple)):
        res_seq: RuntimeTensorOrValue = mx.array(data)
        return res_seq
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
      logging.getLogger(__name__).debug("mlx.core.array conversion failed: %s", exc)
    return data

  def get_tiered_examples(self) -> dict[str, str]:
    """Return MLX idiomatic examples used for validity testing.

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
      "tier4_qwen3-vl": """import mlx.core as mx
import mlx.nn as nn

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(nn.Module):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        cfg = self.config
        seq_len = hidden_states.shape[0]

        hidden_states = mx.reshape(
            hidden_states,
            [seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        )
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1))

        out = self.proj(hidden_states)
        return mx.reshape(out, [seq_len, cfg.hidden_size])
""",
    }

  def get_device_syntax(self, device_type: str, device_index: str | None = None) -> str:
    """Return device constructor syntax.

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
    """Check if default device is GPU.

    Note: MLX Unified Memory doesn't have strict 'is_available' but we check backend.

    Returns:
        str: Code string.

    """
    return "mx.default_device() == mx.gpu"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """MLX usually uses implicit state, but if explicit mode is requested,.

    return 'pass' as split logic differs significantly.

    Args:
        rng_var (str): The RNG variable to split.
        key_var (str): The key variable to store the split result.

    Returns:
        str: "pass".

    """
    return "pass"

  def get_serialization_imports(self) -> list[str]:
    """Return imports for serialization.

    Returns:
        List[str]: Imports.

    """
    return ["import mlx.core as mx"]

  def apply_wiring(self, snapshot: dict[str, dict[str, JsonValue]]) -> None:
    """Override/Patches snapshot items that cannot be statically defined.

    Args:
        snapshot: Snapshotdict.

    """
    pass

  def get_doc_url(self, api_name: str) -> str | None:
    """Generate documentation URL for MLX APIs using autosummary pattern.

    Args:
        api_name: Fully qualified API string.

    Returns:
        Optional[str]: URL.

    """
    return f"https://ml-explore.github.io/mlx/build/html/python/_autosummary/{api_name}.html"
