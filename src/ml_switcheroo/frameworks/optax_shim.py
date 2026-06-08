"""Optax Scanner Logic.

This module provides introspection for the Optax library to power the "Ghost Protocol"
discovery. Optax uses a functional API where optimizers and losses are functions
returning named tuples or callables, rather than Classes.

Capabilities:
1.  Scans `optax.losses` for loss functions.
2.  Scans root `optax` for optimizer factory functions.
3.  Filters internal utilities to provide clean Abstract Standard candidates.
"""

try:
  import optax
except Exception:
  optax = None


class OptaxScanner:
  """Helper to inspect Optax APIs for the discovery system."""
