from enum import Enum


class SupportedEngine(str, Enum):
  """
  Backends supported by the ml_switcheroo transpiler.
  Values match the library names used in semantic JSONs.
  """

  TORCH = "torch"
  JAX = "jax"
  NUMPY = "numpy"
  TENSORFLOW = "tensorflow"
  MLX = "mlx"


class SemanticTier(str, Enum):
  """
  Categorization of API operations to distinct JSON lookup files.
  """

  ARRAY_API = "array"  # Pure Math (semantics/k_array_api.json)
  NEURAL = "neural"  # Layers/Stateful (semantics/k_neural_net.json)
  EXTRAS = "extras"  # Framework Utilities (semantics/k_framework_extras.json)
