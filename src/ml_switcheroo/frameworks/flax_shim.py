"""
Flax Scanner Logic.

Handles introspection of Flax Linen layers (the standard object-oriented layer API for JAX).
Scans `flax.linen` for Modules.
"""

import inspect
from typing import List

try:
  import flax.linen
except ImportError:
  flax = None

from ml_switcheroo.core.ghost import GhostInspector, GhostRef


class FlaxScanner:
  """
  Helper to inspect Flax Linen Modules.
  """

  @staticmethod
  def is_available() -> bool:
    return flax is not None

  @staticmethod
  def scan_layers() -> List[GhostRef]:
    """
    Scans flax.linen for Module subclasses.
    """
    if not flax:
      return []

    results = []

    # Iterate over flax.linen members
    for name, obj in inspect.getmembers(flax.linen):
      # Skip internals
      if name.startswith("_"):
        continue

      if inspect.isclass(obj):
        # Ensure it is a valid Module but not the base itself
        # Defensively check issubclass to avoid crashing on weird imports
        try:
          if issubclass(obj, flax.linen.Module) and name != "Module":
            # Check for standard layer naming conventions to avoid utilities
            # PascalCase usually implies layer
            if name[0].isupper():
              ref = GhostInspector.inspect(obj, f"flax.linen.{name}")
              results.append(ref)
        except TypeError:
          pass

    return results
