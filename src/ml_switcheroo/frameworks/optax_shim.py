"""
Optax Scanner Logic.

Optax provides optimization and loss functions for the JAX ecosystem.
Unlike PyTorch (Classes), Optax uses functional interfaces (factory functions
that return GradientTransformations or loss value calculators).
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
  Helper to inspect Optax APIs.
  """

  @staticmethod
  def is_available() -> bool:
    return optax is not None

  @staticmethod
  def scan_optimizers() -> List[GhostRef]:
    """
    Scans internal Optax registry or public API for optimizers.
    Optax optimizers are functions returning GradientTransformation.
    """
    if not optax:
      return []

    results = []

    # Common optimizer names in Optax
    # We maintain a list to avoid picking up utilities
    known_optimizers = {"adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb", "lion", "novograd", "fromage"}

    for name, obj in inspect.getmembers(optax):
      # Skip internals
      if name.startswith("_"):
        continue

      # Heuristic: Must be a function or a Chain (class-like in some versions)
      # We check if name matches known list or ends with 'optimizer'
      is_known = name in known_optimizers
      is_suffixed = name.endswith("_optimizer")

      if (inspect.isfunction(obj) or inspect.isclass(obj)) and (is_known or is_suffixed):
        # Capture the function signature (e.g. adam(learning_rate, b1, ...))
        ref = GhostInspector.inspect(obj, f"optax.{name}")
        results.append(ref)

    return results

  @staticmethod
  def scan_losses() -> List[GhostRef]:
    """
    Scans optax.losses submodule.
    """
    if not optax or not hasattr(optax, "losses"):
      return []

    results = []
    for name, obj in inspect.getmembers(optax.losses):
      if name.startswith("_"):
        continue

      # Losses in Optax are functions returning arrays/scalars
      if inspect.isfunction(obj):
        # Heuristic: ends with loss or error
        name_lower = name.lower()
        if "loss" in name_lower or "error" in name_lower or "entropy" in name_lower:
          ref = GhostInspector.inspect(obj, f"optax.losses.{name}")
          results.append(ref)

    return results
