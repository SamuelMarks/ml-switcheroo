"""
Core Discovery mechanism for Auto-Inference of APIs.

This module provides the ``SimulatedReflection`` engine, which enables the
system to dynamically locate API endpoints in a target framework based on
an abstract operation name. It is used by the ``define`` CLI command to
resolve `api: "infer"` placeholders in ODL (Operation Definition Language) files.
"""

import importlib
import inspect
import difflib
import logging
from typing import Optional, List, Tuple
from ml_switcheroo.frameworks import get_adapter

# Configure local logger
logger = logging.getLogger(__name__)


class SimulatedReflection:
  """
  Scans a target framework to infer the API path for a given operation name.

  It utilizes the ``FrameworkAdapter.search_modules`` to limit the search space
  and applies both exact normalization matching and fuzzy string matching to
  find the best candidate.
  """

  def __init__(self, framework: str):
    """
    Initializes the reflection engine.

    Args:
        framework (str): The key of the framework to inspect (e.g. 'torch').
    """
    self.framework = framework
    self.adapter = get_adapter(framework)

    # Determine modules to scan. Use Adapter config if available, else default to root.
    if self.adapter and hasattr(self.adapter, "search_modules"):
      self.search_modules = self.adapter.search_modules
    else:
      self.search_modules = [framework]

  def discover(self, op_name: str) -> Optional[str]:
    """
    Attempts to locate the fully qualified API path for an operation.

    Strategy:

    1.  **Normalization**: Converts input and candidates to lowercase, stripping underscores.
    2.  **Exact Match**: Iterates through search modules. If a normalized match is found, returns immediately.
    3.  **Fuzzy Match**: If no exact match is found, collects all candidates and uses `difflib`
        to find the closest string match.

    Args:
        op_name (str): The abstract operation name (e.g. "LogSoftmax", "abs").

    Returns:
        Optional[str]: The fully qualified API path (e.g. "torch.nn.functional.log_softmax")
        or None if no confident match is found.
    """
    normalized_target = self._normalize(op_name)

    # 1. Exact / Normalized Match (Priority)
    for mod_name in self.search_modules:
      try:
        mod = importlib.import_module(mod_name)
      except ImportError:
        logger.debug(f"Could not import {mod_name} during discovery.")
        continue

      for name, _ in inspect.getmembers(mod):
        if name.startswith("_"):
          continue

        if self._normalize(name) == normalized_target:
          return f"{mod_name}.{name}"

    # 2. Fuzzy Match (Fallback)
    candidates: List[Tuple[str, str]] = []

    for mod_name in self.search_modules:
      try:
        mod = importlib.import_module(mod_name)
        # Collect all public members
        candidates.extend(
          [(name, f"{mod_name}.{name}") for name, _ in inspect.getmembers(mod) if not name.startswith("_")]
        )
      except ImportError:
        continue

    if not candidates:
      return None

    # Extract just names for diffing
    candidate_names = [c[0] for c in candidates]

    # Use a high cutoff (0.6) to avoid nonsensical matches
    matches = difflib.get_close_matches(op_name, candidate_names, n=1, cutoff=0.6)

    if matches:
      best_match_name = matches[0]
      # Retrieve the full path for the best match
      for name, path in candidates:
        if name == best_match_name:
          return path

    return None

  def _normalize(self, s: str) -> str:
    """
    Normalizes a string for comparison (lowercase, no underscores).

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    return s.lower().replace("_", "")
