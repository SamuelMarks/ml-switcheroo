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

from ml_switcheroo.testing.adapters import get_adapter


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
               Example: {"x": "Array", "axis": "int | None"}.

    Returns:
        Dict[str, Any]: Randomized inputs ready for Framework adaptation.
    """
    kwargs = {}
    # Decide on a consistent shape for this batch of inputs
    # to ensure compatibility for binary ops (e.g. x + y)
    base_shape = self._get_random_shape()
    hints = hints or {}

    for p in params:
      hint = hints.get(p)

      # Strategy 1: Explicit Type Hint
      if hint and hint != "Any":
        # Spec reader might return complex strings like "int | None"
        try:
          val = self._generate_from_hint(hint, base_shape, depth=0)
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

  def _generate_from_hint(self, type_str: str, base_shape: Tuple[int, ...], depth: int) -> Any:
    """
    Parses a type string and generates conforming data.

    Handles:
    - Primitives: int, float, bool, str
    - Structures: Optional[T], Tuple[T, ...], List[T], Dict[K, V]
    - Unions: A | B
    - Domain Objects: Array, Tensor, Dtype

    Args:
        type_str: The type annotation string (e.g., "Optional[int]").
        base_shape: The active array shape for context.
        depth: Current recursion depth to prevent stack overflow.

    Returns:
        Random data matching the type.
    """
    type_str = type_str.strip()

    # Circuit breaker for recursion
    if depth > self.max_depth:
      return self._get_fallback_base_value(type_str, base_shape)

    # 1. Unions (A | B)
    # Note: This split is naive and doesn't handle nested pipes within brackets well.
    # However, spec_reader typically simplifies complex nesting.
    if "|" in type_str:
      # Check if pipe is top level (not inside brackets)
      if self._is_pipe_top_level(type_str):
        options = [o.strip() for o in type_str.split("|")]
        chosen = random.choice(options)
        return self._generate_from_hint(chosen, base_shape, depth + 1)

    # 2. Optional (Optional[T])
    match_opt = re.match(r"^Optional\[(.*)]$", type_str)
    if match_opt:
      if random.random() < 0.2:
        return None
      return self._generate_from_hint(match_opt.group(1), base_shape, depth + 1)

    # 3. Tuple (Tuple[T, U] or Tuple[T, ...])
    match_tup = re.match(r"^Tuple\[(.*)]$", type_str)
    if match_tup:
      inner = match_tup.group(1)
      # Tuple[int, int] vs Tuple[int, ...]
      if "..." in inner:
        # Variadic tuple: generate random length (1-3) of the first type
        elem_type = inner.split(",")[0].strip()
        length = random.randint(1, 3)
        return tuple(self._generate_from_hint(elem_type, base_shape, depth + 1) for _ in range(length))
      else:
        # Fixed tuple
        sub_types = self._split_outside_brackets(inner)
        return tuple(self._generate_from_hint(t, base_shape, depth + 1) for t in sub_types)

    # 4. List/Sequence (List[T])
    match_list = re.match(r"^(List|Sequence)\[(.*)]$", type_str)
    if match_list:
      inner = match_list.group(2)
      length = random.randint(1, 3)
      return [self._generate_from_hint(inner, base_shape, depth + 1) for _ in range(length)]

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
          k = self._generate_from_hint(key_type, base_shape, depth + 1)
          v = self._generate_from_hint(val_type, base_shape, depth + 1)
          # Ensure key is hashable
          try:
            data[k] = v
          except TypeError:
            # Fallback for unhashable generated key (like array)
            safe_k = str(k)
            data[safe_k] = v
        return data
      return {}

    # 6. None
    if type_str in ["None", "NoneType"]:
      return None

    # 7. Arrays / Tensors
    if type_str in ["Array", "Tensor", "np.ndarray"]:
      # Default to float array
      return self._generate_array("float", base_shape)

    # 8. Primitives
    if type_str in ["int", "integer"]:
      return self._generate_scalar_int()
    if type_str in ["float", "number"]:
      return self._generate_scalar_float()
    if type_str == "bool":
      return bool(random.getrandbits(1))
    if type_str == "str":
      return "test_val_" + str(random.randint(0, 100))

    # 9. Dtype objects
    if "dtype" in type_str.lower():
      # Return a valid numpy dtype object
      return random.choice([np.float32, np.int32, np.bool_])

    # Fallback to float scalar if unknown non-array type, or array if generic
    return self._generate_array("float", base_shape)

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
    if type_str in ["Array", "Tensor"]:
      return np.zeros(base_shape, dtype=np.float32)
    if type_str.startswith("List") or type_str.startswith("Sequence"):
      return []
    if type_str.startswith("Tuple"):
      return ()
    if type_str.startswith("Dict") or type_str.startswith("Mapping"):
      return {}
    return None  # Safe default

  def _generate_by_heuristic(self, name: str, base_shape: Tuple[int, ...]) -> Any:
    """Fallback generation based on argument name patterns."""
    name_lower = name.lower()

    # Contextual Integers (Axis, Dim)
    if name_lower in ["axis", "dim"]:
      rank = len(base_shape)
      # Return valid axis for shape
      return random.randint(0, rank - 1)

    if name_lower in ["keepdim", "keepdims"]:
      return random.choice([True, False])

    if name_lower in ["shape", "size"]:
      return base_shape

    # Dtype heuristics
    heuristic_type = self._guess_dtype_by_name(name)
    if heuristic_type in ["bool", "int", "float"]:
      # Determine if we want array or scalar based on name?
      # Usually input args 'x', 'y' are arrays. 'alpha', 'beta' are scalars.
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
    # Float
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
