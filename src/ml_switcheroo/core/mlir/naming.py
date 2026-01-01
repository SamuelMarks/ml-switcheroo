"""
Naming Context for MLIR Generator.

This module provides the `NamingContext` class, which manages variable name
resolution during MLIR-to-Python translation. It handles:
- Mapping SSA IDs (e.g. `%0`) to valid Python identifiers (e.g. `_0`).
- Avoiding reserved Python keywords.
- Handling hint-based naming (e.g. `%flat` -> `_flat`).
- Collision resolution for duplicate names in the same scope.
"""

from typing import Dict, Optional
import re


class NamingContext:
  """
  Tracks mapping between MLIR SSA IDs and Python variable names.
  Ensures generated names are valid identifiers and do not collide.
  """

  def __init__(self) -> None:
    """Initialize map and reserved keywords."""
    # Map SSA Identifier (e.g. "%0") -> Python Identifier (e.g. "v0")
    self._map: Dict[str, str] = {}
    # Track used python names to prevent collision
    self._used_names: Dict[str, bool] = {}
    # Reserved python keywords + return to avoid collision logic
    # Removed 'self' to allow clean reconstruction of methods
    self._reserved = {
      "return",
      "def",
      "class",
      "if",
      "else",
      "for",
      "import",
      "from",
      "as",
      "with",
    }

  def register(self, ssa_name: str, hint: Optional[str] = None) -> str:
    """
    Assigns a valid Python name to an SSA value.

    Naming Strategy:

    1. If hint provided: Use hint (cleaned).
    2. If SSA ID (e.g. %res): Use prefix + ID body (e.g. _res).
    3. Heuristic: Strip trailing numeric counters from SSA hints if base is unique
       (e.g., %self0 -> self, %x5 -> x).

    Args:
        ssa_name: The MLIR variable name (e.g. "%0", "%arg0").
        hint: Optional string to guide naming (e.g. "flatten" from torch.flatten).

    Returns:
        str: The resolved Python identifier string.
    """
    base = "v"

    if hint:
      # Clean start/chars
      clean = hint.lstrip("%").replace(".", "_")

      # Heuristic: If hint ends in digits (e.g. self0), try stripping them
      # to recover original name 'self', unless it collides.
      match = re.match(r"([a-zA-Z_]+)\d+$", clean)
      if match:
        candidate = match.group(1)
        # Only use stripped name if it's safe (not reserved/used)
        if candidate not in self._used_names and candidate not in self._reserved:
          base = candidate
        else:
          base = clean
      else:
        base = clean

      # Semantic Hint Prefixing
      # If explicit hint is provided (e.g. "flatten"), prepend "_" if it doesn't already have one,
      # to denote internal/generated variable usage typically.
      # If it was an SSA reference (e.g. %self), base will be 'self'.

      # We distinguish explicit semantic hints from SSA hints by checking structure
      if not ssa_name.endswith(hint) and not hint.startswith("_") and not ssa_name.startswith(f"%{hint}"):
        # This implies 'hint' came from op type, not ssa ID
        if not base.startswith("_"):
          base = f"_{base}"

    elif ssa_name.startswith("%"):
      base = "_" + ssa_name[1:]

    py_name = base

    # Fallback/Collision Resolution
    if not py_name.isidentifier() or py_name in self._reserved or py_name in self._used_names:
      # Collision or invalid: Try prepending underscore
      if not py_name.startswith("_"):
        attempt = f"_{py_name}"
      else:
        attempt = py_name

      # If still used or invalid, fallback to indexed v
      if attempt in self._used_names or not attempt.isidentifier():
        # Simple collision resolution logic
        count = 0
        while True:
          # Clean might be undefined if hint was None
          prefix = "v"
          if hint:
            prefix = hint.lstrip("%").replace(".", "_")
            if not prefix.startswith("_"):
              prefix = f"_{prefix}"

          attempt = f"{prefix}_{count}"
          if attempt not in self._used_names:
            break
          count += 1

        py_name = attempt
      else:
        py_name = attempt

    self._map[ssa_name] = py_name
    self._used_names[py_name] = True
    return py_name

  def lookup(self, ssa_name: str) -> str:
    """
    Retrieves Python name for SSA ID.

    Args:
        ssa_name: The MLIR variable name.

    Returns:
        The mapped Python name, or safe fallback if not registered.
    """
    if ssa_name in self._map:
      return self._map[ssa_name]

    # Global references (functions, classes) often stored with @ prefix in Emitter
    if ssa_name.startswith("@"):
      return ssa_name[1:]

    # Fallback replacing % with _ if somehow not registered
    return ssa_name.replace("%", "_")
