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

from typing import List, Tuple, Optional, Dict, Any
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import register_framework, StructuralTraits, PluginTraits, StandardMap, ImportConfig
from ml_switcheroo.frameworks.loader import load_definitions


from ml_switcheroo.frameworks.mlx_io import MlxIOMixin

np: Any
try:
  import numpy as _np

  np = _np
except Exception:  # pragma: no cover
  np = None  # pragma: no cover


@register_framework("mlx")
class MLXAdapter(MlxIOMixin):
  """Adapter for Apple MLX (Silicon-optimized tensor framework)."""

  display_name: str = "Apple MLX"
  inherits_from: Optional[str] = None
  ui_priority: int = 50

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Default alias for core array operations: ``import mlx.core as mx``.

    Returns:
        Tuple[str, str]: ("mlx.core", "mx").

    """
    return "mlx.core", "mx"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
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
  def test_config(self) -> Dict[str, str]:
    """Templates for generating physical verification tests.

    Returns:
        Dict[str, str]: Templates.

    """
    return {
      "import": "import mlx.core as mx\nimport numpy as np",
      "convert_input": "mx.array({np_var})",
      "to_numpy": "np.array({res_var})",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Imports for harness.

    Returns:
        List[str]: Empty list.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Initialization code.

    Returns:
        str: Empty string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns code to convert MLX arrays (which have .tolist()) to NumPy.

    Returns:
        str: Python logic for conversion.

    """
    return "if hasattr(obj, 'tolist'): return np.array(obj.tolist())"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Returns supported semantic tiers (Array, Neural, Extras).

    Returns:
        List[SemanticTier]: Supported Tiers.

    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """Implicit RNG arguments.

    Returns:
        List[str]: Empty.

    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Defines structural rewriting rules (Classes, Methods, Init).

    Updated to strip 'rngs' argument coming from Flax NNX, as MLX
    handles initialization statefully/eagerly.

    Returns:
        StructuralTraits: Config object.

    """
    return StructuralTraits(
      module_base="mlx.nn.Module", forward_method="__call__", requires_super_init=True, auto_strip_magic_args=True
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Plugin behavior configuration.

    Returns:
        PluginTraits: Config object.

    """
    return PluginTraits(  # pragma: no cover
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Static definitions for MLX mappings.


    Loaded dynamically from `frameworks/definitions/mlx.json`.

    Returns:
        Dict[str, StandardMap]: Definitions map.

    """
    return load_definitions("mlx")

  @property
  def rng_seed_methods(self) -> List[str]:
    """Returns list of global seed setters.

    Returns:
        List[str]: Method names.

    """
    return ["seed", "random.seed"]

  def convert(self, data: Any) -> Any:
    """Converts input data (NumPy/List) to MLX Tensor for verification.

    Args:
        data (Any): Input.

    Returns:
        Any: MLX Array or original.

    """
    try:
      import mlx.core as mx

      if isinstance(data, (np.ndarray, list, tuple, np.generic)):  # pragma: no cover
        return mx.array(data)  # pragma: no cover
    except Exception:
      pass
    return data

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns MLX idiomatic examples used for validity testing.

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

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Returns device constructor syntax.

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

    Returns:
        str: "pass".

    """
    return "pass"  # pragma: no cover

  def get_serialization_imports(self) -> List[str]:
    """Returns imports for serialization.

    Returns:
        List[str]: Imports.

    """
    return ["import mlx.core as mx"]  # pragma: no cover

  def apply_wiring(self, snapshot: Any) -> Any:
    """Overrides/Patches snapshot items that cannot be statically defined.

    Args:
        snapshot: Snapshotdict.

    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Generates documentation URL for MLX APIs using autosummary pattern.

    Args:
        api_name: Fully qualified API string.

    Returns:
        Optional[str]: URL.

    """
    return f"https://ml-explore.github.io/mlx/build/html/python/_autosummary/{api_name}.html"
