"""Framework Adapters Package.

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
from ml_switcheroo.frameworks.base import FrameworkAdapter, register_framework, get_adapter, available_frameworks

_EXCLUDED_MODULES = {"base", "__init__", "common", "optax_shim", "loader"}


def _auto_register_adapters() -> None:
  """Scans the current package for modules and imports them.

  Importing the module triggers the @register_framework decorator defined
  within the adapter implementation, populating the global registry.
  """
  for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name in _EXCLUDED_MODULES:
      continue
    try:
      importlib.import_module(f".{module_name}", package=__name__)
    except Exception as e:
      logging.warning(f"⚠️  Failed to load framework module '{module_name}': {e}")


_auto_register_adapters()
__all__ = ["FrameworkAdapter", "available_frameworks", "get_adapter", "register_framework"]
