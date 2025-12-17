"""
Input Generation Engine for Semantics Verification.

This module provides the `InputFuzzer`, responsible for creating randomized
NumPy arrays, scalars, and complex container structures to feed into Framework A
and Framework B during behavioral verification tests.

It uses a dual-strategy approach:

1.  **Type Hints (Semantic Validated)**:
    If explicit type information is available (e.g., from Spec files), the fuzzer
    parses these strings using a mini-grammar to generate compliant data.

    **Supported Grammar:**
    - Primitives: `int`, `float`, `bool`, `str`
    - Structural: `List[T]`, `Tuple[T, ...]`, `Dict[K, V]`, `Optional[T]`
    - Unions: `int | float`
    - Domain: `Array`, `Tensor`, `dtype`
    - **Symbolic Shapes**: `Array['B', 'N']` (Constraint Solving for consistent dimensions)

2.  **Heuristics (Fallback)**:
    If no hints are available, it infers data types from argument names.
    - `mask`, `condition` -> `bool`
    - `idx`, `axis`, `dim` -> `int` within bounds
    - `x`, `y`, `input` -> `Array (float32)`

Safety:
    Implements `MAX_RECURSION_DEPTH` to prevent infinite loops when generating
    recursive or deeply nested types.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml_switcheroo.frameworks import get_adapter


class InputFuzzer:
  """
  Generates dummy inputs (Arrays, Scalars, Containers) for equivalence testing.

  Attributes:
      _seed_shape (Optional[Tuple[int, ...]]): If set, forces generation
          of arrays with this specific shape in fallback mode.
      max_depth (int): Maximum recursion depth for nested container types.
  """

  MAX_RECURSION_DEPTH = 3

  def __init__(self, seed_shape: Optional[Tuple[int, ...]] = None):
    """
    Initializes the Fuzzer.

    Args:
        seed_shape: If provided, heuristic array generation defaults to this shape.
    """
    self._seed_shape = seed_shape
    self.max_depth = self.MAX_RECURSION_DEPTH

  def generate_inputs(self, params: List[str], hints: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Creates a dictionary of `{arg_name: value}`.

    Resolves symbolic dimensions across the entire parameter set. For example,
    if hints are `{'x': "Array['N']", 'y': "Array['N']"}`, both arrays will
    have matching lengths.

    Args:
        params: List of argument names to generate (e.g. `['x', 'axis']`).
        hints: Dictionary of `{arg_name: type_string}` derived from Spec.

    Returns:
        Dict[str, Any]: Randomized inputs ready for Framework adaptation.
    """
    kwargs: Dict[str, Any] = {}
    # Context to resolve symbolic dimensions like 'B', 'N' across arguments
    symbol_map: Dict[str, int] = {}

    # Decide on a consistent base shape for heuristics fallback
    base_shape = self._get_random_shape()
    hints = hints or {}

    for p in params:
      hint = hints.get(p)

      # Strategy 1: Explicit Type Hint
      if hint and hint != "Any":
        try:
          val = self._generate_from_hint(hint, base_shape, depth=0, symbol_map=symbol_map)
          kwargs[p] = val
          continue
        except Exception:
          # If parsing fails or hits recursion limit, fall back to heuristics
          pass

      # Strategy 2: Heuristic Matching based on Name
      kwargs[p] = self._generate_by_heuristic(p, base_shape)

    return kwargs

  def adapt_to_framework(self, kwargs: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """
    Converts Numpy inputs to framework-specific tensor types.

    Delegates to registered adapters (e.g., `TorchAdapter`, `JaxAdapter`).

    Args:
        kwargs: Input dictionary with Numpy values.
        framework: Key of the framework (e.g., "torch", "jax").

    Returns:
        Dict with framework-specific tensors.
    """
    adapter = get_adapter(framework)

    # If no adapter found, return pure numpy/python objects (Pass-through)
    if not adapter:
      return kwargs

    converted = {}
    for k, v in kwargs.items():
      try:
        converted[k] = adapter.convert(v)
      except Exception:
        # If conversion logic fails, keep original
        converted[k] = v

    return converted

  def _get_random_shape(self) -> Tuple[int, ...]:
    """
    Selects a random rank (1-4) and random dimensions (2-5).
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
    symbol_map: Dict[str, int],
  ) -> Any:
    """
    Parses a type string and generates conforming data via recursion.
    """
    if depth > self.max_depth:
      return self._get_fallback_base_value(type_str, base_shape)

    type_str = type_str.strip()

    # 1. Unions (A | B) from Python 3.10 syntax
    if "|" in type_str and self._is_pipe_top_level(type_str):
      options = [o.strip() for o in type_str.split("|")]
      chosen = random.choice(options)
      return self._generate_from_hint(chosen, base_shape, depth + 1, symbol_map)

    # 2. Optional (Optional[T])
    match_opt = re.match(r"^Optional\[(.*)\]$", type_str)
    if match_opt:
      if random.random() < 0.2:
        return None
      return self._generate_from_hint(match_opt.group(1), base_shape, depth + 1, symbol_map)

    # 3. Tuple (Tuple[T, ...] or Tuple[T, U])
    match_tup = re.match(r"^Tuple\[(.*)\]$", type_str)
    if match_tup:
      inner = match_tup.group(1)
      if "..." in inner:
        # Variadic Tuple
        elem_type = inner.split(",")[0].strip()
        length = random.randint(1, 3)
        return tuple(self._generate_from_hint(elem_type, base_shape, depth + 1, symbol_map) for _ in range(length))
      else:
        # Fixed Tuple
        sub_types = self._split_outside_brackets(inner)
        return tuple(self._generate_from_hint(t, base_shape, depth + 1, symbol_map) for t in sub_types)

    # 4. List/Sequence (List[T])
    match_list = re.match(r"^(List|Sequence)\[(.*)\]$", type_str)
    if match_list:
      inner = match_list.group(2)
      length = random.randint(1, 3)
      return [self._generate_from_hint(inner, base_shape, depth + 1, symbol_map) for _ in range(length)]

    # 5. Dict/Mapping (Dict[K, V])
    match_dict = re.match(r"^(Dict|Mapping)\[(.*)\]$", type_str)
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
          # Convert key to string if unhashable (e.g. array)
          if isinstance(k, (list, dict, np.ndarray, np.generic)):
            k = str(k)
          data[k] = v
        return data
      return {}

    # 6. Nulls
    if type_str in ["None", "NoneType"]:
      return None

    # 7. Symbolic Arrays / Tensors: Array['B', 'N']
    match_sym = re.match(r"^(Array|Tensor|np\.ndarray)\[(.*)\]$", type_str)
    if match_sym:
      dims_str = match_sym.group(2)
      shape = self._resolve_symbolic_shape(dims_str, symbol_map)
      return self._generate_array("float", shape)

    # 8. Generic Arrays
    if type_str in ["Array", "Tensor", "np.ndarray"]:
      return self._generate_array("float", base_shape)

    # 9. Primitives
    if type_str in ["int", "integer"]:
      return self._generate_scalar_int()
    if type_str in ["float", "number"]:
      return self._generate_scalar_float()
    if type_str == "bool":
      return bool(random.getrandbits(1))
    if type_str in ["str", "string"]:
      return "val_" + str(random.randint(0, 100))

    # 10. Dtype objects
    if "dtype" in type_str.lower():
      return random.choice([np.float32, np.int32, np.float64, np.bool_])

    # Fallback for unknown strings
    return self._generate_array("float", base_shape)

  def _resolve_symbolic_shape(self, dims_str: str, symbol_map: Dict[str, int]) -> Tuple[int, ...]:
    """
    Parses dimension strings like "'B', 32" or "N, M".
    Resolves symbols against the `symbol_map`.

    Args:
        dims_str: Comma-separated dimension tokens.
        symbol_map: Mutable map of symbol->size.

    Returns:
        Tuple representing the concrete shape.
    """
    shape = []
    raw_dims = [d.strip() for d in dims_str.split(",")]

    for dim in raw_dims:
      # Clean possible quotes
      clean = dim.replace("'", "").replace('"', "")
      if not clean:
        continue

      # Check if it's a fixed integer literal (e.g., Array[3, 'C'])
      try:
        val = int(clean)
        shape.append(val)
        continue
      except ValueError:
        pass

      # Validate symbol is a valid identifier or simple char
      if not clean.isidentifier() and not clean.replace("_", "").isalnum():
        # Fallback random for complex expressions we don't parse (e.g. N+1)
        shape.append(random.randint(2, 6))
        continue

      # It's a symbol
      if clean not in symbol_map:
        symbol_map[clean] = random.randint(2, 6)
      shape.append(symbol_map[clean])

    return tuple(shape)

  def _get_fallback_base_value(self, type_str: str, base_shape: Tuple[int, ...]) -> Any:
    """Returns minimal valid value to terminate recursion."""
    if type_str == "bool":
      return False
    if type_str in ["int", "integer"]:
      return 0
    if type_str in ["float", "number"]:
      return 0.0
    if type_str == "str":
      return ""
    if type_str.startswith(("Array", "Tensor")):
      return np.zeros(base_shape, dtype=np.float32)
    if type_str.startswith(("List", "Sequence")):
      return []
    if type_str.startswith("Tuple"):
      return ()
    if type_str.startswith(("Dict", "Mapping")):
      return {}
    return None

  def _generate_by_heuristic(self, name: str, base_shape: Tuple[int, ...]) -> Any:
    """Fallback generation when no type hint is provided."""
    name_lower = name.lower()

    if name_lower in ["axis", "dim"]:
      # Ensure index is within rank of base_shape
      rank = len(base_shape)
      return random.randint(0, max(0, rank - 1))

    if name_lower in ["keepdim", "keepdims"]:
      return random.choice([True, False])

    if name_lower in ["shape", "size"]:
      return base_shape

    heuristic_type = self._guess_dtype_by_name(name)
    if heuristic_type == "bool":
      return self._generate_array("bool", base_shape)
    if heuristic_type == "int":
      # Scalars check
      if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):
        return self._generate_scalar_int()
      return self._generate_array("int", base_shape)

    # Floats
    if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):
      return self._generate_scalar_float()

    return self._generate_array("float", base_shape)

  def _generate_array(self, type_lbl: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Generates random numpy array."""
    if type_lbl == "bool":
      return np.random.randint(0, 2, size=shape).astype(bool)
    if type_lbl == "int":
      return np.random.randint(-10, 10, size=shape).astype(np.int32)
    # Float32 default
    return np.random.randn(*shape).astype(np.float32)

  def _generate_scalar_int(self) -> int:
    return random.randint(-5, 5)

  def _generate_scalar_float(self) -> float:
    return random.random()

  def _guess_dtype_by_name(self, name: str) -> str:
    """Simple name matching heuristic."""
    name_lower = name.lower()
    if any(x in name_lower for x in ["mask", "condition", "is_", "p_val"]):
      return "bool"
    if any(x in name_lower for x in ["idx", "index", "indices", "k", "n_", "count"]):
      return "int"
    return "float"

  def _is_pipe_top_level(self, text: str) -> bool:
    """Checks if a pipe `|` exists outside of brackets `[]` for Union detection."""
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
    """Splits string by comma, ensuring correctness for nested generics."""
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
