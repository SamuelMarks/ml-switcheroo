"""
Type Hint Parser and Recursive Generation Logic.

This module processes string type hints (e.g. `List[Array['N']]`) and
generates conforming data structures.
"""

import re
import random
import numpy as np
from typing import Any, Dict, Tuple

from ml_switcheroo.testing.fuzzer.generators import (
  generate_array,
  generate_scalar_int,
  generate_scalar_float,
  generate_fake_callable,
)
from ml_switcheroo.testing.fuzzer.utils import (
  is_pipe_top_level,
  split_outside_brackets,
  resolve_symbolic_shape,
  adjust_shape_rank,
)


def get_fallback_base_value(type_str: str, base_shape: Tuple[int, ...]) -> Any:
  """
  Returns a minimal valid value to terminate recursion when depth limit is reached.
  """
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

  # Feature: Callable Support
  if type_str.startswith("Callable") or type_str in ["func", "function"]:
    return generate_fake_callable()

  return None


def generate_from_hint(
  type_str: str,
  base_shape: Tuple[int, ...],
  depth: int,
  max_depth: int,
  symbol_map: Dict[str, int],
  constraints: Dict[str, Any] = None,
) -> Any:
  """
  Recursively parses a type string and generates conforming data.

  Handles structural types (List, Tuple, Dict, Union, Optional) and
  content types (Array, int, float, dtype, Callable).

  Args:
      type_str (str): The type hint string to parse.
      base_shape (Tuple[int]): Default shape helper.
      depth (int): Current recursion depth.
      max_depth (int): Limit for recursion.
      symbol_map (Dict): Shared context for dimension symbols ('N').
      constraints (Dict): Semantic constraints (min, max, options).

  Returns:
      Any: Generated data structure.
  """
  if depth > max_depth:
    return get_fallback_base_value(type_str, base_shape)

  constrs = constraints or {}

  # Options Override: if explicit options provided, pick one
  if "options" in constrs:
    return random.choice(constrs["options"])

  type_str = type_str.strip()

  # 1. Unions (A | B) from Python 3.10 syntax
  if "|" in type_str and is_pipe_top_level(type_str):
    options = [o.strip() for o in type_str.split("|")]
    chosen = random.choice(options)
    return generate_from_hint(chosen, base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 2. Optional (Optional[T])
  match_opt = re.match(r"^Optional\[(.*)\]$", type_str)
  if match_opt:
    if random.random() < 0.2:
      return None
    return generate_from_hint(match_opt.group(1), base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 3. Tuple (Tuple[T, ...] or Tuple[T, U])
  match_tup = re.match(r"^Tuple\[(.*)\]$", type_str)
  if match_tup:
    # Constraints generally apply to the tuple contents if uniform, or ignored.
    inner = match_tup.group(1)
    if "..." in inner:
      # Variadic Tuple
      elem_type = inner.split(",")[0].strip()
      length = random.randint(1, 3)
      return tuple(
        generate_from_hint(elem_type, base_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length)
      )
    else:
      # Fixed Tuple
      sub_types = split_outside_brackets(inner)
      return tuple(generate_from_hint(t, base_shape, depth + 1, max_depth, symbol_map, constraints) for t in sub_types)

  # 4. List/Sequence (List[T])
  match_list = re.match(r"^(List|Sequence)\[(.*)\]$", type_str)
  if match_list:
    inner = match_list.group(2)
    length = random.randint(1, 3)
    return [generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length)]

  # 5. Dict/Mapping (Dict[K, V])
  match_dict = re.match(r"^(Dict|Mapping)\[(.*)\]$", type_str)
  if match_dict:
    inner = match_dict.group(2)
    parts = split_outside_brackets(inner)
    if len(parts) == 2:
      key_type, val_type = parts
      length = random.randint(1, 3)
      data = {}
      for _ in range(length):
        k = generate_from_hint(key_type, base_shape, depth + 1, max_depth, symbol_map)
        # Apply value constraints to the Value
        v = generate_from_hint(val_type, base_shape, depth + 1, max_depth, symbol_map, constraints)
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
    shape = resolve_symbolic_shape(dims_str, symbol_map)
    # Resolve rank override if present
    if constrs.get("rank"):
      shape = adjust_shape_rank(shape, constrs["rank"])
    return generate_array("float", shape, constrs)

  # 8. Generic Arrays
  if type_str in ["Array", "Tensor", "np.ndarray"]:
    shape = base_shape
    if constrs.get("rank"):
      shape = adjust_shape_rank(shape, constrs["rank"])
    return generate_array("float", shape, constrs)

  # 9. Callables (New Feature)
  if type_str.startswith("Callable") or type_str in ["func", "function"]:
    return generate_fake_callable(constrs)

  # 10. Primitives
  if type_str in ["int", "integer"]:
    return generate_scalar_int(constrs)
  if type_str in ["float", "number"]:
    return generate_scalar_float(constrs)
  if type_str == "bool":
    return bool(random.getrandbits(1))
  if type_str in ["str", "string"]:
    return "val_" + str(random.randint(0, 100))

  # 11. Dtype objects
  if "dtype" in type_str.lower():
    # Dtype constraints on a dtype object is meta (e.g. valid types)
    return random.choice([np.float32, np.int32, np.float64, np.bool_])

  # Fallback for unknown strings
  return generate_array("float", base_shape, constrs)
