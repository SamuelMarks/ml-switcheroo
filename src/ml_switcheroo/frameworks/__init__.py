"""
Framework Adapters Package.

Automatically discovers and registers framework adapters.
Exposes concrete Adapter classes for testing/typing convenience.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List

from .base import FrameworkAdapter, _ADAPTER_REGISTRY, register_framework, get_adapter

# --- Concrete Imports (Updated for JAX Hierarchy) ---

try:
  from .torch import TorchAdapter
except ImportError:
  TorchAdapter = None

try:
  # Maps generic 'jax' requests to the Core adapter
  from .jax import JaxCoreAdapter as JaxAdapter
except ImportError:
  JaxAdapter = None

try:
  from .flax_nnx import FlaxNNXAdapter
except ImportError:
  FlaxNNXAdapter = None

try:
  from .paxml import PaxmlAdapter
except ImportError:
  PaxmlAdapter = None

try:
  from .numpy import NumpyAdapter
except ImportError:
  NumpyAdapter = None

try:
  from .tensorflow import TensorFlowAdapter
except ImportError:
  TensorFlowAdapter = None

try:
  from .mlx import MLXAdapter
except ImportError:
  MLXAdapter = None

# --- Dynamic Discovery ---
# Scans the directory for new .py files (e.g. 'my_framework.py')
# and imports them to trigger the @register_framework decorator.
_pkg_dir = Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([str(_pkg_dir)]):
  if module_name in ("base", "__init__"):
    continue
  try:
    importlib.import_module(f".{module_name}", package=__name__)
  except Exception as e:
    print(f"⚠️  Failed to load framework module '{module_name}': {e}")


def available_frameworks() -> List[str]:
  """Returns a list of all registered framework keys."""
  return list(_ADAPTER_REGISTRY.keys())


__all__ = [
  "get_adapter",
  "register_framework",
  "FrameworkAdapter",
  "available_frameworks",
  "TorchAdapter",
  "JaxAdapter",
  "FlaxNNXAdapter",
  "NumpyAdapter",
  "TensorFlowAdapter",
  "MLXAdapter",
  "PaxmlAdapter",
]
