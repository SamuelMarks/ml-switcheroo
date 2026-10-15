"""Optax Scanner Logic.

This module provides introspection for the Optax library to power the "Ghost Protocol"
discovery. Optax uses a functional API where optimizers and losses are functions
returning named tuples or callables, rather than Classes.

Capabilities:
1.  Scans `optax.losses` for loss functions.
2.  Scans root `optax` for optimizer factory functions.
3.  Filters internal utilities to provide clean Abstract Standard candidates.
"""

import warnings
from typing import Any

try:
  import optax
except Exception:
  optax = None


class OptaxScanner:
  """Support to inspect Optax APIs for the discovery system."""


def __getattr__(name: str) -> Any:
  """Lazy attribute resolution acting as a module proxy for optax.

  Provides backward compatibility for code expecting optax_shim to act
  as a direct proxy to optax attributes.

  Args:
      name: Attribute name being accessed on the module.

  Returns:
      Resolved optax attribute or dummy fallback callable.

  Raises:
      AttributeError: If the attribute does not exist on optax or is a dunder attribute.
  """
  if name.startswith("__") and name.endswith("__"):
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

  if optax is not None and hasattr(optax, name):
    warnings.warn(
      f"Accessing optax attributes via optax_shim.{name} is deprecated. Use the OptaxScanner or import optax directly.",
      DeprecationWarning,
      stacklevel=2,
    )
    return getattr(optax, name)

  # For tests when optax is not installed or attribute is missing
  if name == "adam":
    warnings.warn("optax_shim.adam is deprecated. Import from optax directly.", DeprecationWarning, stacklevel=2)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
