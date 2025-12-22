"""
Input Generation Engine for Semantics Verification.

This module provides the `InputFuzzer`, responsible for creating randomized
NumPy arrays, scalars, and complex container structures to feed into Framework A
and Framework B during behavioral verification tests.

Updates:
    - Support for `dtype` constraints in array generation (Limitation #2 Fix).
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml_switcheroo.frameworks import get_adapter


class InputFuzzer:
  """
  Generates dummy inputs (Arrays, Scalars, Containers) for equivalence testing.
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

  def generate_inputs(
    self,
    params: List[str],
    hints: Optional[Dict[str, str]] = None,
    constraints: Optional[Dict[str, Dict]] = None,
  ) -> Dict[str, Any]:
    """
    Creates a dictionary of `{arg_name: value}`.

    Resolves symbolic dimensions across the entire parameter set. For example,
    if hints are `{'x': "Array['N']", 'y': "Array['N']"}`, both arrays will
    have matching lengths.

    Args:
        params: List of argument names to generate (e.g. `['x', 'axis']`).
        hints: Dictionary of `{arg_name: type_string}` derived from Spec.
        constraints: Dictionary of `{arg_name: {min, max, options, dtype}}`.

    Returns:
        Dict[str, Any]: Randomized inputs ready for Framework adaptation.
    """
    kwargs: Dict[str, Any] = {}
    # Context to resolve symbolic dimensions like 'B', 'N' across arguments
    symbol_map: Dict[str, int] = {}

    # Decide on a consistent base shape for heuristics fallback
    base_shape = self._get_random_shape()
    hints = hints or {}
    constraints_map = constraints or {}

    for p in params:
      hint = hints.get(p)
      cons = constraints_map.get(p, {})

      # Strategy 1: Explicit Type Hint
      if hint and hint != "Any":
        try:
          val = self._generate_from_hint(hint, base_shape, depth=0, symbol_map=symbol_map, constraints=cons)
          kwargs[p] = val
          continue
        except Exception:
          # If parsing fails, fall back to heuristics... BUT apply constraints if possible?
          # Or better, let recursion handle logic. Exception implies structural failure.
          pass

      # Strategy 2: Heuristic Matching based on Name
      kwargs[p] = self._generate_by_heuristic(p, base_shape, constraints=cons)

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
    constraints: Dict[str, Any] = None,
  ) -> Any:
    """
    Parses a type string and generates conforming data via recursion.
    Applies Semantic Constraints (min, max, options, dtype) to leaf values.
    """
    if depth > self.max_depth:
      return self._get_fallback_base_value(type_str, base_shape)

    constrs = constraints or {}

    # Options Override: if explicit options provided, pick one
    if "options" in constrs:
      return random.choice(constrs["options"])

    type_str = type_str.strip()

    # 1. Unions (A | B) from Python 3.10 syntax
    if "|" in type_str and self._is_pipe_top_level(type_str):
      options = [o.strip() for o in type_str.split("|")]
      chosen = random.choice(options)
      return self._generate_from_hint(chosen, base_shape, depth + 1, symbol_map, constraints)

    # 2. Optional (Optional[T])
    match_opt = re.match(r"^Optional\[(.*)\]$", type_str)
    if match_opt:
      if random.random() < 0.2:
        return None
      return self._generate_from_hint(match_opt.group(1), base_shape, depth + 1, symbol_map, constraints)

    # 3. Tuple (Tuple[T, ...] or Tuple[T, U])
    match_tup = re.match(r"^Tuple\[(.*)\]$", type_str)
    if match_tup:
      # Constraints generally apply to the tuple contents if uniform, or ignored.
      # Passing constraints recursively is naive but safe.
      inner = match_tup.group(1)
      if "..." in inner:
        # Variadic Tuple
        elem_type = inner.split(",")[0].strip()
        length = random.randint(1, 3)
        return tuple(
          self._generate_from_hint(elem_type, base_shape, depth + 1, symbol_map, constraints) for _ in range(length)
        )
      else:
        # Fixed Tuple
        sub_types = self._split_outside_brackets(inner)
        return tuple(self._generate_from_hint(t, base_shape, depth + 1, symbol_map, constraints) for t in sub_types)

    # 4. List/Sequence (List[T])
    match_list = re.match(r"^(List|Sequence)\[(.*)\]$", type_str)
    if match_list:
      inner = match_list.group(2)
      length = random.randint(1, 3)
      return [self._generate_from_hint(inner, base_shape, depth + 1, symbol_map, constraints) for _ in range(length)]

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
          # Apply value constraints to the Value
          v = self._generate_from_hint(val_type, base_shape, depth + 1, symbol_map, constraints)
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
      # Resolve rank override if present
      if constrs.get("rank"):
        shape = self._adjust_shape_rank(shape, constrs["rank"])
      return self._generate_array("float", shape, constrs)

    # 8. Generic Arrays
    if type_str in ["Array", "Tensor", "np.ndarray"]:
      shape = base_shape
      if constrs.get("rank"):
        shape = self._adjust_shape_rank(shape, constrs["rank"])
      return self._generate_array("float", shape, constrs)

    # 9. Primitives
    if type_str in ["int", "integer"]:
      return self._generate_scalar_int(constrs)
    if type_str in ["float", "number"]:
      return self._generate_scalar_float(constrs)
    if type_str == "bool":
      return bool(random.getrandbits(1))
    if type_str in ["str", "string"]:
      return "val_" + str(random.randint(0, 100))

    # 10. Dtype objects
    if "dtype" in type_str.lower():
      # Dtype constraints on a dtype object is meta (e.g. valid types), but implementation is simpler
      return random.choice([np.float32, np.int32, np.float64, np.bool_])

    # Fallback for unknown strings
    return self._generate_array("float", base_shape, constrs)

  def _resolve_symbolic_shape(self, dims_str: str, symbol_map: Dict[str, int]) -> Tuple[int, ...]:
    """
    Parses dimension strings like "'B', 32" or "N, M".
    Resolves symbols against the `symbol_map`.
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

  def _adjust_shape_rank(self, shape: Tuple[int, ...], required_rank: int) -> Tuple[int, ...]:
    """Adjusts shape to match required rank."""
    current_rank = len(shape)
    if current_rank == required_rank:
      return shape
    elif current_rank < required_rank:
      # Pad
      extra = tuple(random.randint(2, 5) for _ in range(required_rank - current_rank))
      return shape + extra
    else:
      # Truncate
      return shape[:required_rank]

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

  def _generate_by_heuristic(self, name: str, base_shape: Tuple[int, ...], constraints: Dict[str, Any] = None) -> Any:
    """Fallback generation when no type hint is provided. Respects constraints."""
    constrs = constraints or {}

    if "options" in constrs:
      return random.choice(constrs["options"])

    name_lower = name.lower()

    if name_lower in ["axis", "dim"]:
      # Ensure index is within rank of base_shape
      rank = len(base_shape)
      return random.randint(0, max(0, rank - 1))

    if name_lower in ["keepdim", "keepdims"]:
      return random.choice([True, False])

    if name_lower in ["shape", "size"]:
      return base_shape

    # Apply Explicit Dtype Requests if present in constraint without type hint
    if constrs.get("dtype"):
      if "int" in constrs["dtype"]:
        return self._generate_array("int", base_shape, constrs)
      if "bool" in constrs["dtype"]:
        return self._generate_array("bool", base_shape, constrs)

    heuristic_type = self._guess_dtype_by_name(name)
    if heuristic_type == "bool":
      return self._generate_array("bool", base_shape, constrs)
    if heuristic_type == "int":
      # Scalars check e.g. alpha
      if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):
        return self._generate_scalar_int(constrs)
      return self._generate_array("int", base_shape, constrs)

    # Floats
    if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):
      return self._generate_scalar_float(constrs)

    return self._generate_array("float", base_shape, constrs)

  def _generate_array(self, type_lbl: str, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
    """Generates random numpy array bounded by constraints."""

    # Resolve bounds
    # Note: numpy works better with standard types (not None)
    min_val = constraints.get("min")
    max_val = constraints.get("max")

    # Limitation #2 Fix: Dtype Support
    dtype_req = constraints.get("dtype")
    explicit_dtype = None
    if dtype_req:
      try:
        explicit_dtype = np.dtype(dtype_req)
      except TypeError:
        pass

    # Use explicit dtype if available to override heuristic label
    # E.g. type_lbl='float' but constraint='int64' -> force int64
    if explicit_dtype:
      if np.issubdtype(explicit_dtype, np.integer):
        type_lbl = "int"
      elif explicit_dtype.kind == "b":
        type_lbl = "bool"
      else:
        type_lbl = "float"

    if type_lbl == "bool":
      return np.random.randint(0, 2, size=shape).astype(bool)

    if type_lbl == "int":
      # Default range [-10, 10]
      low = int(min_val) if min_val is not None else -10
      # High in randint is exclusive, update for inclusivity match
      # If max is None default to 10, else use max+1
      high_bound = int(max_val) if max_val is not None else 10
      high = high_bound + 1

      arr = np.random.randint(low, high, size=shape)
      if explicit_dtype:
        return arr.astype(explicit_dtype)
      return arr.astype(np.int32)

    # Float default
    arr = np.random.randn(*shape)

    # Constraint Clipping logic
    if min_val is not None or max_val is not None:
      if min_val is not None and max_val is not None:
        # Uniform within bounds
        arr = np.random.uniform(min_val, max_val, size=shape)
      else:
        # Clip standard normal
        safe_min = float(min_val) if min_val is not None else -np.inf
        safe_max = float(max_val) if max_val is not None else np.inf

        # If just Min is 0 (Log), use absolute or exp
        if min_val is not None and min_val >= 0:
          arr = np.abs(arr) + min_val

        arr = np.clip(arr, safe_min, safe_max)

    if explicit_dtype:
      return arr.astype(explicit_dtype)

    return arr.astype(np.float32)

  def _generate_scalar_int(self, constraints: Dict[str, Any]) -> int:
    min_v = int(constraints.get("min", -5))
    max_v = int(constraints.get("max", 5))
    return random.randint(min_v, max_v)

  def _generate_scalar_float(self, constraints: Dict[str, Any]) -> float:
    if constraints.get("min") is not None and constraints.get("max") is not None:
      return random.uniform(constraints["min"], constraints["max"])

    val = random.random()
    # Adjust
    if constraints.get("min") is not None:
      val = max(val, constraints["min"])
    if constraints.get("max") is not None:
      val = min(val, constraints["max"])
    return val

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
