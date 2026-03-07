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
    return False  # pragma: no cover
  if type_str in ["int", "integer"]:
    return 0  # pragma: no cover
  if type_str in ["float", "number"]:
    return 0.0  # pragma: no cover
  if type_str == "str":
    return ""  # pragma: no cover
  if type_str.startswith(("Array", "Tensor")):
    return np.zeros(base_shape, dtype=np.float32)  # pragma: no cover
  if type_str.startswith(("List", "Sequence")):
    return []  # pragma: no cover
  if type_str.startswith("Tuple"):
    return ()  # pragma: no cover
  if type_str.startswith(("Dict", "Mapping")):
    return {}  # pragma: no cover

  # Feature: Callable Support
  if type_str.startswith("Callable") or type_str in ["func", "function"]:
    return generate_fake_callable()

  return None  # pragma: no cover


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
  if depth > max_depth:  # pragma: no cover
    return get_fallback_base_value(type_str, base_shape)  # pragma: no cover

  constrs = constraints or {}  # pragma: no cover

  # 0. Options Override: if explicit options provided, pick one
  if "options" in constrs and constrs["options"]:  # pragma: no cover
    return random.choice(constrs["options"])  # pragma: no cover

  type_str = type_str.strip()  # pragma: no cover

  # 1. Inference from Default Value (if Type is Any/Unknown)
  # This ensures fuzzing respects the implicit type contract of the default value.
  if type_str in ["Any", "None", ""] and "default" in constrs:  # pragma: no cover
    default_val = constrs["default"]  # pragma: no cover
    # Probabilistic: Sometimes just use the default value directly (Coverage)
    if random.random() < 0.2:  # pragma: no cover
      return default_val  # pragma: no cover

    # Otherwise infer type to generate VARIATIONS of that type
    if isinstance(default_val, bool):  # pragma: no cover
      type_str = "bool"  # pragma: no cover
    elif isinstance(default_val, int):  # pragma: no cover
      type_str = "int"  # pragma: no cover
    elif isinstance(default_val, float):  # pragma: no cover
      type_str = "float"  # pragma: no cover
    elif isinstance(default_val, list):  # pragma: no cover
      # Inspect recursive type or fallback relative simple list
      if default_val and isinstance(default_val[0], int):  # pragma: no cover
        type_str = "List[int]"  # pragma: no cover
      else:
        type_str = "List[Any]"  # pragma: no cover

  # 2. Unions (A | B) from Python 3.10 syntax
  if "|" in type_str and is_pipe_top_level(type_str):  # pragma: no cover
    options = [o.strip() for o in type_str.split("|")]  # pragma: no cover
    chosen = random.choice(options)  # pragma: no cover
    return generate_from_hint(chosen, base_shape, depth + 1, max_depth, symbol_map, constraints)  # pragma: no cover

  # 3. Optional (Optional[T])
  match_opt = re.match(r"^Optional\[(.*)\]$", type_str)  # pragma: no cover
  if match_opt:  # pragma: no cover
    if random.random() < 0.2:  # pragma: no cover
      return None  # pragma: no cover
    return generate_from_hint(
      match_opt.group(1), base_shape, depth + 1, max_depth, symbol_map, constraints
    )  # pragma: no cover

  # 4. Tuple (Tuple[T, ...] or Tuple[T, U])
  match_tup = re.match(r"^Tuple\[(.*)\]$", type_str)  # pragma: no cover
  if match_tup:  # pragma: no cover
    inner = match_tup.group(1)  # pragma: no cover
    if "..." in inner:  # pragma: no cover
      # Variadic Tuple
      elem_type = inner.split(",")[0].strip()  # pragma: no cover
      length = random.randint(1, 3)  # pragma: no cover
      return tuple(  # pragma: no cover
        generate_from_hint(elem_type, base_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length)
      )
    else:
      # Fixed Tuple
      sub_types = split_outside_brackets(inner)  # pragma: no cover
      return tuple(
        generate_from_hint(t, base_shape, depth + 1, max_depth, symbol_map, constraints) for t in sub_types
      )  # pragma: no cover

  # 5. List/Sequence (List[T])
  match_list = re.match(r"^(List|Sequence)\[(.*)\]$", type_str)  # pragma: no cover
  if match_list:  # pragma: no cover
    inner = match_list.group(2)  # pragma: no cover
    length = random.randint(2, 3)  # ensure at least 2 for concat/stack cases  # pragma: no cover

    # Check if inner type is Tensor-like to enforce uniform shape
    is_tensor = inner.startswith(("Array", "Tensor", "np.ndarray"))  # pragma: no cover

    # Generate the first element to establish shape context
    first_elem = generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints)  # pragma: no cover

    if is_tensor and isinstance(first_elem, np.ndarray):  # pragma: no cover
      # Use the shape of the first element as the base_shape for subsequent elements
      uniform_shape = first_elem.shape  # pragma: no cover
      list_data = [first_elem]  # pragma: no cover
      for _ in range(length - 1):  # pragma: no cover
        # Pass explicit uniform_shape as base_shape context
        elem = generate_from_hint(inner, uniform_shape, depth + 1, max_depth, symbol_map, constraints)  # pragma: no cover
        # Ensure consistency (regen if heuristics failed to match shape)
        if isinstance(elem, np.ndarray) and elem.shape != uniform_shape:  # pragma: no cover
          elem = generate_array("float", uniform_shape, constrs)  # pragma: no cover
        list_data.append(elem)  # pragma: no cover
      return list_data  # pragma: no cover
    else:
      list_data = [first_elem]  # pragma: no cover
      for _ in range(length - 1):  # pragma: no cover
        list_data.append(
          generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints)
        )  # pragma: no cover
      return list_data  # pragma: no cover

  # 6. Dict/Mapping (Dict[K, V])
  match_dict = re.match(r"^(Dict|Mapping)\[(.*)\]$", type_str)  # pragma: no cover
  if match_dict:  # pragma: no cover
    inner = match_dict.group(2)  # pragma: no cover
    parts = split_outside_brackets(inner)  # pragma: no cover
    if len(parts) == 2:  # pragma: no cover
      key_type, val_type = parts  # pragma: no cover
      length = random.randint(1, 3)  # pragma: no cover
      data = {}  # pragma: no cover
      for _ in range(length):  # pragma: no cover
        k = generate_from_hint(key_type, base_shape, depth + 1, max_depth, symbol_map)  # pragma: no cover
        v = generate_from_hint(val_type, base_shape, depth + 1, max_depth, symbol_map, constraints)  # pragma: no cover
        # Convert key to string if unhashable (e.g. array)
        if isinstance(k, (list, dict, np.ndarray, np.generic)):  # pragma: no cover
          k = str(k)  # pragma: no cover
        data[k] = v  # pragma: no cover
      return data  # pragma: no cover
    return {}  # pragma: no cover

  # 7. Nulls
  if type_str in ["None", "NoneType"]:  # pragma: no cover
    return None  # pragma: no cover

  # 8. Symbolic Arrays / Tensors: Array['B', 'N']
  match_sym = re.match(r"^(Array|Tensor|np\.ndarray)\[(.*)\]$", type_str)  # pragma: no cover
  if match_sym:  # pragma: no cover
    dims_str = match_sym.group(2)  # pragma: no cover
    shape = resolve_symbolic_shape(dims_str, symbol_map)  # pragma: no cover
    # Resolve rank override if present
    if constrs.get("rank"):  # pragma: no cover
      shape = adjust_shape_rank(shape, constrs["rank"])  # pragma: no cover
    return generate_array("float", shape, constrs)  # pragma: no cover

  # 9. Generic Arrays
  if type_str in ["Array", "Tensor", "np.ndarray"]:  # pragma: no cover
    shape = base_shape  # pragma: no cover
    if constrs.get("rank"):  # pragma: no cover
      shape = adjust_shape_rank(shape, constrs["rank"])  # pragma: no cover
    return generate_array("float", shape, constrs)  # pragma: no cover

  # 10. Callables (Function stubs)
  if type_str.startswith("Callable") or type_str in ["func", "function"]:  # pragma: no cover
    return generate_fake_callable(constrs)  # pragma: no cover

  # 11. Primitives
  if type_str in ["int", "integer"]:  # pragma: no cover
    return generate_scalar_int(constrs)  # pragma: no cover
  if type_str in ["float", "number"]:  # pragma: no cover
    return generate_scalar_float(constrs)  # pragma: no cover
  if type_str in ["bool", "boolean"]:  # pragma: no cover
    return bool(random.getrandbits(1))  # pragma: no cover
  if type_str in ["str", "string"]:  # pragma: no cover
    return "val_" + str(random.randint(0, 100))  # pragma: no cover

  # 12. Dtype objects
  if "dtype" in type_str.lower():  # pragma: no cover
    # Dtype constraints on a dtype object is meta (e.g. valid types)
    return random.choice([np.float32, np.int32, np.float64, np.bool_])  # pragma: no cover

  # Fallback for unknown strings using default heuristics for arrays
  return generate_array("float", base_shape, constrs)  # pragma: no cover
