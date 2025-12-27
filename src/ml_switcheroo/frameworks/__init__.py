"""
Framework Adapters Package.

Automatically discovers and registers framework adapters by scanning this
directory for modules. This allows for "Zero-Edit" extensibility: simply drop
a new adapter file (e.g., `tinygrad.py`) into this folder, and it will be
automatically registered.

This module exposes the registry helpers (`get_adapter`, `available_frameworks`)
but relies on the side-effects of importing submodules to populate the
internal `_ADAPTER_REGISTRY`.
"""

import importlib
import pkgutil
import logging
from pathlib import Path
from typing import List, Optional

from ml_switcheroo.frameworks.base import (
  FrameworkAdapter,
  _ADAPTER_REGISTRY,
  register_framework,
  get_adapter,
)

# Modules to exclude from the automatic adapter registration scan.
# These are infrastructure or helper modules, not Framework Adapters.
_EXCLUDED_MODULES = {"base", "__init__", "common", "optax_shim"}


def _auto_register_adapters() -> None:
  """
  Scans the current directory for .py files and imports them.

  Importing the module triggers the @register_framework decorator defined
  within the adapter implementation, populating the global registry.
  """
  pkg_path = str(Path(__file__).parent)

  for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
    if module_name in _EXCLUDED_MODULES:
      continue

    try:
      # Dynamically import the module relative to this package.
      # e.g., import ml_switcheroo.frameworks.torch
      importlib.import_module(f".{module_name}", package=__name__)
    except Exception as e:
      # We log exceptions but do not crash. This ensures that one broken
      # plugin/adapter does not prevent the entire engine from starting.
      logging.warning(f"⚠️  Failed to load framework module '{module_name}': {e}. This framework will not be available.")


# Execute discovery on import
_auto_register_adapters()


def available_frameworks() -> List[str]:
  """
  Returns a list of all registered framework keys.

  The keys are populated dynamically. Usage example:
  >>> if "tinygrad" in available_frameworks():
  ...     print("TinyGrad is supported!")

  Returns:
      List[str]: A list of identifier strings (e.g. ['torch', 'jax']).
  """
  # Keys correspond to the string passed to @register_framework("key")
  return list(_ADAPTER_REGISTRY.keys())


__all__ = [
  "FrameworkAdapter",
  "available_frameworks",
  "get_adapter",
  "register_framework",
]
