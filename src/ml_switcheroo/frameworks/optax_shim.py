"""
Optax Scanner Logic.

This module provides introspection for the Optax library to power the "Ghost Protocol"
discovery. Optax uses a functional API where optimizers and losses are functions
returning named tuples or callables, rather than Classes.

Capabilities:
1.  Scans `optax.losses` for loss functions.
2.  Scans root `optax` for optimizer factory functions.
3.  Filters internal utilities to provide clean Abstract Standard candidates.
"""

import inspect
from typing import List

try:
  import optax
except ImportError:
  optax = None

from ml_switcheroo.core.ghost import GhostInspector, GhostRef


class OptaxScanner:
  """
  Helper to inspect Optax APIs for the discovery system.
  """

  @staticmethod
  def scan_optimizers() -> List[GhostRef]:
    """
    Scans the Optax public API for optimizer functions.

    Optax optimizers are typically factory functions (e.g., `adam`, `sgd`)
    that return a `GradientTransformation`.
    """
    if not optax:
      return []

    results = []

    # List of known optimizer names to prioritize logic
    # We also use heuristic matching for unknown extensions
    known_optimizers = {
      "adam",
      "adamw",
      "sgd",
      "rmsprop",
      "adagrad",
      "lamb",
      "lion",
      "novograd",
      "fromage",
      "yogi",
      "adabelief",
    }

    for name, obj in inspect.getmembers(optax):
      # Skip internals and private members
      if name.startswith("_"):
        continue

      # Heuristic: Must be a function (factories) or class (some aliases)
      if inspect.isfunction(obj) or inspect.isclass(obj):
        # Check 1: Is in known list
        is_known = name.lower() in known_optimizers

        # Check 2: Ends with 'optimizer' suffix convention
        is_suffixed = name.endswith("_optimizer")

        # Check 3: Is it an alias to a known transformation in Alias mode?
        # (Skipped for simplicity, relying on name match)

        if is_known or is_suffixed:
          try:
            # Capture signature (e.g. adam(learning_rate, b1, ...))
            ref = GhostInspector.inspect(obj, f"optax.{name}")
            results.append(ref)
          except Exception:
            pass

    return results

  @staticmethod
  def scan_losses() -> List[GhostRef]:
    """
    Scans `optax.losses` for loss functions.
    """
    if not optax or not hasattr(optax, "losses"):
      return []

    results = []

    for name, obj in inspect.getmembers(optax.losses):
      if name.startswith("_"):
        continue

      # Optax losses are pure functions returning arrays/scalars
      if inspect.isfunction(obj):
        # Heuristic: Name contains 'loss', 'error', or 'entropy'
        name_lower = name.lower()
        if "loss" in name_lower or "error" in name_lower or "entropy" in name_lower:
          try:
            ref = GhostInspector.inspect(obj, f"optax.losses.{name}")
            results.append(ref)
          except Exception:
            pass

    return results
