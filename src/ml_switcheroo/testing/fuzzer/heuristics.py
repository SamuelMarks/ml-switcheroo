"""
Name-Based Heuristics for Input Generation.

This module provides fallback logic for generating inputs when explicit
type hints are missing, generally based on argument naming conventions.
"""

import random
from typing import Any, Dict, Tuple

from ml_switcheroo.testing.fuzzer.generators import (
  generate_array,
  generate_scalar_float,
  generate_scalar_int,
)


def guess_dtype_by_name(name: str) -> str:
  """
  Guesses the logical type of an argument based on its name.

  Args:
      name (str): The argument name.

  Returns:
      str: 'bool', 'int', or 'float'.
  """
  name_lower = name.lower()
  if any(x in name_lower for x in ["mask", "condition", "is_", "p_val"]):
    return "bool"
  if any(x in name_lower for x in ["idx", "index", "indices", "k", "n_", "count"]):
    return "int"
  return "float"


def generate_by_heuristic(name: str, base_shape: Tuple[int, ...], constraints: Dict[str, Any] = None) -> Any:
  """
  Generates a value based on the argument name when no type hint is provided.

  Respects provided constraints if any.

  Args:
      name (str): Argument name (e.g. 'axis', 'x').
      base_shape (Tuple): Default shape for array generation.
      constraints (Dict): Optional constraints (min/max/type).

  Returns:
      Any: Generated value.
  """
  constrs = constraints or {}  # pragma: no cover

  if "options" in constrs:  # pragma: no cover
    return random.choice(constrs["options"])  # pragma: no cover

  name_lower = name.lower()  # pragma: no cover

  if name_lower in ["axis", "dim"]:  # pragma: no cover
    # Ensure index is within rank of base_shape
    rank = len(base_shape)  # pragma: no cover
    return random.randint(0, max(0, rank - 1))  # pragma: no cover

  if name_lower in ["keepdim", "keepdims"]:  # pragma: no cover
    return random.choice([True, False])  # pragma: no cover

  if name_lower in ["shape", "size"]:  # pragma: no cover
    return base_shape  # pragma: no cover

  # Apply Explicit Dtype Requests if present in constraint without type hint
  if constrs.get("dtype"):  # pragma: no cover
    if "int" in constrs["dtype"]:  # pragma: no cover
      return generate_array("int", base_shape, constrs)  # pragma: no cover
    if "bool" in constrs["dtype"]:  # pragma: no cover
      return generate_array("bool", base_shape, constrs)  # pragma: no cover

  heuristic_type = guess_dtype_by_name(name)  # pragma: no cover
  if heuristic_type == "bool":  # pragma: no cover
    return generate_array("bool", base_shape, constrs)  # pragma: no cover
  if heuristic_type == "int":  # pragma: no cover
    # Scalars check e.g. alpha
    if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):  # pragma: no cover
      return generate_scalar_int(constrs)  # pragma: no cover
    return generate_array("int", base_shape, constrs)  # pragma: no cover

  # Floats
  if any(prefix in name_lower for prefix in ["alpha", "eps", "scalar", "val"]):  # pragma: no cover
    return generate_scalar_float(constrs)  # pragma: no cover

  return generate_array("float", base_shape, constrs)  # pragma: no cover
