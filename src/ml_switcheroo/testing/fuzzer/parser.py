"""
Type Hint Parser and Recursive Generation Logic.

This module processes string type hints (e.g. `List[Array['N']]`) and
generates conforming data structures for runtime fuzzing.

It includes logic to:
1.  Parse nested generic types.
2.  Resolve symbolic shape constraints.
3.  Respect semantic constraints (`min`, `max`, `options`).
4.  **Infer generation strategy from default values** when explicit hints are erased.
"""

import re
import random
import numpy as np
from typing import Any, Dict, Tuple, Optional

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

  Args:
      type_str: The target type hint.
      base_shape: Default shape for array fallbacks.

  Returns:
      Safe fallback value (0, empty list, etc).
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
  constraints: Optional[Dict[str, Any]] = None,
) -> Any:
  """
  Recursively parses a type string and generates conforming data.

  If type hints are generic ("Any"), it attempts to infer the type logic
  from the `default` value provided in constraints.

  Args:
      type_str: The type hint string to parse.
      base_shape: Default shape helper.
      depth: Current recursion depth.
      max_depth: Limit for recursion.
      symbol_map: Shared context for dimension symbols ('N').
      constraints: Semantic constraints (min, max, options, default).

  Returns:
      Generated data structure.
  """
  if depth > max_depth:
    return get_fallback_base_value(type_str, base_shape)

  constrs = constraints or {}

  # 0. Options Override: if explicit options provided, pick one
  if "options" in constrs and constrs["options"]:
    return random.choice(constrs["options"])

  type_str = type_str.strip()

  # 1. Inference from Default Value (if Type is Any/Unknown)
  # This ensures fuzzing respects the implicit type contract of the default value.
  if type_str in ["Any", "None", ""] and "default" in constrs:
    default_val = constrs["default"]
    # Probabilistic: Sometimes just use the default value directly (Coverage)
    if random.random() < 0.2:
      return default_val

    # Otherwise infer type to generate VARIATIONS of that type
    if isinstance(default_val, bool):
      type_str = "bool"
    elif isinstance(default_val, int):
      type_str = "int"
    elif isinstance(default_val, float):
      type_str = "float"
    elif isinstance(default_val, list):
      # Inspect recursive type or fallback relative simple list
      if default_val and isinstance(default_val[0], int):
        type_str = "List[int]"
      else:
        type_str = "List[Any]"

  # 2. Unions (A | B) from Python 3.10 syntax
  if "|" in type_str and is_pipe_top_level(type_str):
    options = [o.strip() for o in type_str.split("|")]
    chosen = random.choice(options)
    return generate_from_hint(chosen, base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 3. Optional (Optional[T])
  match_opt = re.match(r"^Optional\[(.*)\]$", type_str)
  if match_opt:
    if random.random() < 0.2:
      return None
    return generate_from_hint(match_opt.group(1), base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 4. Tuple (Tuple[T, ...] or Tuple[T, U])
  match_tup = re.match(r"^Tuple\[(.*)\]$", type_str)
  if match_tup:
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

  # 5. List/Sequence (List[T])
  match_list = re.match(r"^(List|Sequence)\[(.*)\]$", type_str)
  if match_list:
    inner = match_list.group(2)
    length = random.randint(2, 3)  # ensure at least 2 for concat/stack cases

    # Check if inner type is Tensor-like to enforce uniform shape
    is_tensor = inner.startswith(("Array", "Tensor", "np.ndarray"))

    # Generate the first element to establish shape context
    first_elem = generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints)

    if is_tensor and isinstance(first_elem, np.ndarray):
      # Use the shape of the first element as the base_shape for subsequent elements
      uniform_shape = first_elem.shape
      list_data = [first_elem]
      for _ in range(length - 1):
        # Pass explicit uniform_shape as base_shape context
        elem = generate_from_hint(inner, uniform_shape, depth + 1, max_depth, symbol_map, constraints)
        # Ensure consistency (regen if heuristics failed to match shape)
        if isinstance(elem, np.ndarray) and elem.shape != uniform_shape:
          elem = generate_array("float", uniform_shape, constrs)
        list_data.append(elem)
      return list_data
    else:
      list_data = [first_elem]
      for _ in range(length - 1):
        list_data.append(generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints))
      return list_data

  # 6. Dict/Mapping (Dict[K, V])
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
        v = generate_from_hint(val_type, base_shape, depth + 1, max_depth, symbol_map, constraints)
        # Convert key to string if unhashable (e.g. array)
        if isinstance(k, (list, dict, np.ndarray, np.generic)):
          k = str(k)
        data[k] = v
      return data
    return {}

  # 7. Nulls
  if type_str in ["None", "NoneType"]:
    return None

  # 8. Symbolic Arrays / Tensors: Array['B', 'N']
  match_sym = re.match(r"^(Array|Tensor|np\.ndarray)\[(.*)\]$", type_str)
  if match_sym:
    dims_str = match_sym.group(2)
    shape = resolve_symbolic_shape(dims_str, symbol_map)
    # Resolve rank override if present
    if constrs.get("rank"):
      shape = adjust_shape_rank(shape, constrs["rank"])
    return generate_array("float", shape, constrs)

  # 9. Generic Arrays
  if type_str in ["Array", "Tensor", "np.ndarray"]:
    shape = base_shape
    if constrs.get("rank"):
      shape = adjust_shape_rank(shape, constrs["rank"])
    return generate_array("float", shape, constrs)

  # 10. Callables (Function stubs)
  if type_str.startswith("Callable") or type_str in ["func", "function"]:
    return generate_fake_callable(constrs)

  # 11. Primitives
  if type_str in ["int", "integer"]:
    return generate_scalar_int(constrs)
  if type_str in ["float", "number"]:
    return generate_scalar_float(constrs)
  if type_str in ["bool", "boolean"]:
    return bool(random.getrandbits(1))
  if type_str in ["str", "string"]:
    return "val_" + str(random.randint(0, 100))

  # 12. Dtype objects
  if "dtype" in type_str.lower():
    # Dtype constraints on a dtype object is meta (e.g. valid types)
    return random.choice([np.float32, np.int32, np.float64, np.bool_])

  # Fallback for unknown strings using default heuristics for arrays
  return generate_array("float", base_shape, constrs)
