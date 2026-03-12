"""
Plugins Package.

Automatically discovers and registers all plugin modules within this package.
This ensures that adding a new file (e.g., `my_new_op.py`) automatically
registers its hooks without manual edits to this file.
"""

import pkgutil
import importlib
from pathlib import Path

# scan the current directory for modules
_pkg_dir = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(_pkg_dir)]):
  # Skip potential future protected modules if necessary
  if module_name.startswith("_") or "utils" in module_name or "helpers" in module_name:
    continue

  try:
    importlib.import_module(f".{module_name}", package=__name__)
  except Exception as e:
    # We log but continue, ensuring one bad plugin doesn't break the engine
    print(f"⚠️  Failed to auto-load plugin '{module_name}': {e}")
