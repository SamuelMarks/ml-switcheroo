"""
Input Generation Engine for Semantics Verification.

This module provides the `InputFuzzer`, responsible for creating randomized
NumPy arrays, scalars, and complex container structures to feed into Framework A
and Framework B during behavioral verification tests.

## Generation Strategies

It uses a dual-strategy approach:

1.  **Type Hints (Semantic Validated)**:
    If `spec_reader` has extracted explicit type information from the spec files,
    the fuzzer parses these strings using a mini-grammar to generate compliant data.

    **Supported Grammar:**
    - Primitives: `int`, `float`, `bool`, `str`
    - Structural: `List[T]`, `Tuple[T, ...]`, `Dict[K, V]`, `Optional[T]`
    - Unions: `int | float`
    - Domain: `Array`, `Tensor`, `dtype`
    - **Symbolic Shapes (New)**: `Array['B', 'N']` (Constraint Solving)

2.  **Heuristics (Fallback)**:
    If no hints are available in the spec, it infers data types from argument names.
    - `mask`, `condition` -> `bool`
    - `idx`, `axis`, `dim` -> `int`
    - `x`, `y`, `input` -> `Array (float32)`

## Safety

The fuzzer implements `MAX_RECURSION_DEPTH` to prevent infinite loops when generating
recursive or deeply nested types (e.g. `List[List[List[...]]]`).
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml_switcheroo.frameworks import get_adapter


class InputFuzzer:
  """
  Generates dummy inputs (Arrays, Scalars, Containers) for equivalence testing.

  Prioritizes explicit Type Hints from the Knowledge Base. Falls back to
  name-based heuristics if hints are missing or ambiguous (e.g., 'Any').

  Attributes:
      _seed_shape (Optional[Tuple[int, ...]]): If set, forces generation
          of arrays with this specific shape.
      max_depth (int): Maximum recursion depth for nested container types.
  """

  MAX_RECURSION_DEPTH = 3

  def __init__(self, seed_shape: Optional[Tuple[int, ...]] = None):
    """
    Initializes the Fuzzer.

    Args:
        seed_shape: If provided, all generated arrays will share this shape.
                    If None, a random shape is chosen per `generate_inputs` call.
    """
    self._seed_shape = seed_shape
    self.max_depth = self.MAX_RECURSION_DEPTH

  def generate_inputs(self, params: List[str], hints: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Creates a dictionary of {arg_name: value}.

    Args:
        params: List of argument names [x, y, axis].
        hints: Dictionary of {arg_name: type_string} derived from Spec.
               Example: {"x": "Array['B', 'T']", "axis": "int"}.

    Returns:
        Dict[str, Any]: Randomized inputs ready for Framework adaptation.
    """
    kwargs = {}
    # Context for resolving symbolic dimensions across arguments in this call
    symbol_map: Dict[str, int] = {}

    # Decide on a consistent base shape for this batch of inputs fallback
    base_shape = self._get_random_shape()
    hints = hints or {}

    for p in params:
      hint = hints.get(p)

      # Strategy 1: Explicit Type Hint
      if hint and hint != "Any":
        # Spec reader might return complex strings like "int | None"
        try:
          val = self._generate_from_hint(hint, base_shape, depth=0, symbol_map=symbol_map)
          kwargs[p] = val
          continue
        except Exception:
          # If parsing fails, fall back silently to heuristics
          pass

      # Strategy 2: Heuristic Matching based on Name
      kwargs[p] = self._generate_by_heuristic(p, base_shape)

    return kwargs

  def adapt_to_framework(self, kwargs: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """
    Converts Numpy inputs to framework-specific tensor types.
    Delegates to registered adapters in `ml_switcheroo.testing.adapters`.

    Args:
        kwargs: Input dictionary with Numpy values.
        framework: "torch", "jax", "tensorflow", etc.

    Returns:
        Dict with framework-specific tensors.
    """
    adapter = get_adapter(framework)

    # If no adapter found, return pure numpy/python objects (Pass-through)
    if not adapter:
      return kwargs

    converted = {}
    for k, v in kwargs.items():
      converted[k] = adapter.convert(v)

    return converted

  def _get_random_shape(self) -> Tuple[int, ...]:
    """
    Selects a random rank (1-4) and random dimensions (2-5).

    Returns:
        Tuple of integers representing shape.
    """
    if self._seed_shape:
      return self._seed_shape

    rank = random.randint(1, 4)
    return tuple(random.randint(2, 5) for _ in range(rank))

  def _generate_from_hint(
    self,
    type_str: str,
    base_shape: Tuple[int, ...],
    depth: int,
    symbol_map: Optional[Dict[str, int]] = None,
  ) -> Any:
    """
    Parses a type string and generates conforming data.

    Args:
        type_str: The type annotation string (e.g., "Optional[int]").
        base_shape: The active array shape for context.
        depth: Current recursion depth to prevent stack overflow.
        symbol_map: Dictionary mapping symbolic dimension names to integers.

    Returns:
        Random data matching the type.
    """
    if symbol_map is None:
      # Fallback if called directly without symbol context (rare)
      symbol_map = {}

    type_str = type_str.strip()

    # Circuit breaker for recursion
    if depth > self.max_depth:
      return self._get_fallback_base_value(type_str, base_shape)

    # 1. Unions (A | B)
    if "|" in type_str and self._is_pipe_top_level(type_str):
      options = [o.strip() for o in type_str.split("|")]
      chosen = random.choice(options)
      return self._generate_from_hint(chosen, base_shape, depth + 1, symbol_map)

    # 2. Optional (Optional[T])
    match_opt = re.match(r"^Optional\[(.*)]$", type_str)
    if match_opt:
      if random.random() < 0.2:
        return None
      return self._generate_from_hint(match_opt.group(1), base_shape, depth + 1, symbol_map)

    # 3. Tuple (Tuple[T, U] or Tuple[T, ...])
    match_tup = re.match(r"^Tuple\[(.*)]$", type_str)
    if match_tup:
      inner = match_tup.group(1)
      if "..." in inner:
        elem_type = inner.split(",")[0].strip()
        length = random.randint(1, 3)
        return tuple(self._generate_from_hint(elem_type, base_shape, depth + 1, symbol_map) for _ in range(length))
      else:
        sub_types = self._split_outside_brackets(inner)
        return tuple(self._generate_from_hint(t, base_shape, depth + 1, symbol_map) for t in sub_types)

    # 4. List/Sequence (List[T])
    match_list = re.match(r"^(List|Sequence)\[(.*)]$", type_str)
    if match_list:
      inner = match_list.group(2)
      length = random.randint(1, 3)
      return [self._generate_from_hint(inner, base_shape, depth + 1, symbol_map) for _ in range(length)]

    # 5. Dict/Mapping (Dict[K, V])
    match_dict = re.match(r"^(Dict|Mapping)\[(.*)]$", type_str)
    if match_dict:
      inner = match_dict.group(2)
      parts = self._split_outside_brackets(inner)
      if len(parts) == 2:
        key_type, val_type = parts
        length = random.randint(1, 3)
        data = {}
        for _ in range(length):
          k = self._generate_from_hint(key_type, base_shape, depth + 1, symbol_map)
          v = self._generate_from_hint(val_type, base_shape, depth + 1, symbol_map)
          try:
            data[k] = v
          except TypeError:
            data[str(k)] = v
        return data
      return {}

    # 6. None
    if type_str in ["None", "NoneType"]:
      return None

    # 7. Symbolic Arrays / Tensors: Array['B', 'N']
    match_sym = re.match(r"^(Array|Tensor|np\.ndarray)\[(.*)]$", type_str)
    if match_sym:
      dims_str = match_sym.group(2)
      shape = self._resolve_symbolic_shape(dims_str, symbol_map)
      return self._generate_array("float", shape)

    # 8. Generic Arrays / Tensors
    if type_str in ["Array", "Tensor", "np.ndarray"]:
      return self._generate_array("float", base_shape)

    # 9. Primitives
    if type_str in ["int", "integer"]:
      return self._generate_scalar_int()
    if type_str in ["float", "number"]:
      return self._generate_scalar_float()
    if type_str == "bool":
      return bool(random.getrandbits(1))
    if type_str == "str":
      return "test_val_" + str(random.randint(0, 100))

    # 10. Dtype objects
    if "dtype" in type_str.lower():
      return random.choice([np.float32, np.int32, np.bool_])

    return self._generate_array("float", base_shape)

  def _resolve_symbolic_shape(self, dims_str: str, symbol_map: Dict[str, int]) -> Tuple[int, ...]:
    """
    Parses dimensions string like "'B', 32" or "N, M".
    Resolves symbols using the provided context map.
    """
    shape = []
    # Naive split by comma is usually sufficient for dimensions
    raw_dims = [d.strip() for d in dims_str.split(",")]

    for dim in raw_dims:
      # Clean quotes
      clean = dim.replace("'", "").replace('"', "")
      if not clean:
        continue

      # Try explicit integer
      try:
        val = int(clean)
        shape.append(val)
        continue
      except ValueError:
        pass

      # It's a symbol
      if clean not in symbol_map:
        symbol_map[clean] = random.randint(2, 6)
      shape.append(symbol_map[clean])

    return tuple(shape)

  def _get_fallback_base_value(self, type_str: str, base_shape: Tuple[int, ...]):
    """Returns a minimal value for recursion termination."""
    if type_str == "bool":
      return False
    if type_str in ["int", "integer"]:
      return 0
    if type_str in ["float", "number"]:
      return 0.0
    if type_str == "str":
      return ""
    if type_str.startswith("Array") or type_str.startswith("Tensor"):
      return np.zeros(base_shape, dtype=np.float32)
    if type_str.startswith("List") or type_str.startswith("Sequence"):
      return []
    if type_str.startswith("Tuple"):
      return ()
    if type_str.startswith("Dict") or type_str.startswith("Mapping"):
      return {}
    return None

  def _generate_by_heuristic(self, name: str, base_shape: Tuple[int, ...]) -> Any:
    """Fallback generation based on argument name patterns."""
    name_lower = name.lower()

    if name_lower in ["axis", "dim"]:
      rank = len(base_shape)
      return random.randint(0, max(0, rank - 1))

    if name_lower in ["keepdim", "keepdims"]:
      return random.choice([True, False])

    if name_lower in ["shape", "size"]:
      return base_shape

    heuristic_type = self._guess_dtype_by_name(name)
    if heuristic_type in ["bool", "int", "float"]:
      if any(prefix in name_lower for prefix in ["alpha", "beta", "scalar", "eps"]):
        if heuristic_type == "int":
          return self._generate_scalar_int()
        return self._generate_scalar_float()

      return self._generate_array(heuristic_type, base_shape)

    return self._generate_array("float", base_shape)

  def _generate_array(self, type_lbl: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Generates a random numpy array of specific type."""
    if type_lbl == "bool":
      return np.random.randint(0, 2, size=shape).astype(bool)
    if type_lbl == "int":
      return np.random.randint(-10, 10, size=shape).astype(np.int32)
    return np.random.randn(*shape).astype(np.float32)

  def _generate_scalar_int(self) -> int:
    return random.randint(-5, 5)

  def _generate_scalar_float(self) -> float:
    return random.random()

  def _guess_dtype_by_name(self, name: str) -> str:
    """Determines probable dtype based on naming conventions."""
    name_lower = name.lower()
    if any(x in name_lower for x in ["mask", "condition", "is_", "p_val"]):
      return "bool"
    if any(x in name_lower for x in ["idx", "index", "indices", "k", "n_"]):
      return "int"
    return "float"

  def _is_pipe_top_level(self, text: str) -> bool:
    """Checks if a pipe | exists outside of brackets []."""
    depth = 0
    for char in text:
      if char == "[":
        depth += 1
      elif char == "]":
        depth -= 1
      elif char == "|" and depth == 0:
        return True
    return False

  def _split_outside_brackets(self, text: str) -> List[str]:
    """Splits string by comma, ignoring commas inside brackets."""
    parts = []
    current = []
    depth = 0
    for char in text:
      if char == "[":
        depth += 1
        current.append(char)
      elif char == "]":
        depth -= 1
        current.append(char)
      elif char == "," and depth == 0:
        parts.append("".join(current).strip())
        current = []
      else:
        current.append(char)
    if current:
      parts.append("".join(current).strip())
    return parts
