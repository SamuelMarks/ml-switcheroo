"""
Framework Adapters Package.

Automatically discovers and registers framework adapters defined in this directory.
To add support for a new framework, simply add a .py file here with a class
decorated via @register_framework("name").
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type, List

from .base import FrameworkAdapter, _ADAPTER_REGISTRY, register_framework, get_adapter

# Dynamic Discovery
# 1. Iterate over all files in this package directory
_pkg_dir = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(_pkg_dir)]):
  # 2. Skip base/special files
  if module_name in ("base", "__init__"):
    continue

  # 3. Import the module (Executing the @register_framework decorators inside)
  try:
    importlib.import_module(f".{module_name}", package=__name__)
  except Exception as e:
    print(f"⚠️  Failed to load framework module '{module_name}': {e}")


def available_frameworks() -> List[str]:
  """Returns a list of all registered framework keys."""
  return list(_ADAPTER_REGISTRY.keys())


__all__ = ["get_adapter", "_ADAPTER_REGISTRY", "register_framework", "FrameworkAdapter", "available_frameworks"]
