"""
Facade for Framework Adapters.
Delegates to the modular `frameworks` package.
"""

from typing import Type, Optional, Any

# Import the new package which triggers auto-registration
from ml_switcheroo.frameworks import _ADAPTER_REGISTRY, get_adapter, register_framework, FrameworkAdapter

# Re-export concrete classes if needed for type checking/imports
try:
  from ml_switcheroo.frameworks.torch import TorchAdapter
  from ml_switcheroo.frameworks.jax import JaxAdapter
  from ml_switcheroo.frameworks.numpy import NumpyAdapter
  from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
  from ml_switcheroo.frameworks.mlx import MLXAdapter
  from ml_switcheroo.frameworks.paxml import PaxmlAdapter
except ImportError:
  pass


def register_adapter(name: str, cls: Optional[Type] = None):
  """
  Shim for register_framework to support legacy test calls.

  Usage:
      1. @register_adapter("name")
      2. register_adapter("name", Class)
  """
  # Case 1: Decorator usage
  if cls is None:
    return register_framework(name)

  # Case 2: Functional usage (legacy tests)
  _ADAPTER_REGISTRY[name] = cls
