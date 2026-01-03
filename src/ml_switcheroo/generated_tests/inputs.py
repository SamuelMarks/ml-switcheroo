"""
Input Value Code Generation.

This module handles the logic for generating Python *source code strings* that
instantiate test data explicitly in the generated verification scripts.
It parses argument definitions and emits code like `random.randint(1, 5)`
or `np.random.randn(...)`.

Features:
- Infers generation logic from explicit types.
- **Default Value Inference**: Uses default values to guess types when explicit
  hints are missing (e.g. `default=True` implies boolean generation).
- **Constraint Inference**: Uses `min`/`max` to govern random ranges.
"""

import random
from typing import Union, Dict, Any, Optional


def parse_arg_def(arg: Union[str, tuple, dict]) -> Dict[str, Any]:
  """
  Normalizes a heterogeneous argument definition into a standard dictionary.

  Extracts `default`, `min`, `max` to help downstream inference.
  Defaults un-typed arguments to 'Array' to ensure gradient checks in
  test generators correctly identify implicit tensors.

  Args:
      arg: An argument definition. Can be a string ("x"), a tuple ("x", "int"),
           or a dictionary (ODL schema).

  Returns:
      Normalized dict with keys 'name', 'type', 'default', and constraints.
  """
  if isinstance(arg, str):
    # Default string args to Array (Tensor)
    return {"name": arg, "type": "Array"}
  elif isinstance(arg, (tuple, list)) and len(arg) == 2:
    return {"name": arg[0], "type": arg[1]}
  elif isinstance(arg, dict):
    # Shallow copy to avoid mutation
    normalized = arg.copy()
    # Default missing types to Array (Tensor)
    if "type" not in normalized or normalized["type"] == "Any":
      # If default is provided, we might infer type later, but for the
      # generator's differentiable check, we default to Array unless
      # scalar Default implies otherwise.
      default_val = normalized.get("default")
      if default_val is not None:
        if isinstance(default_val, bool):
          normalized["type"] = "bool"
        elif isinstance(default_val, int):
          normalized["type"] = "int"
        elif isinstance(default_val, float):
          normalized["type"] = "float"
        else:
          normalized["type"] = "Array"
      else:
        normalized["type"] = "Array"
    return normalized
  return {"name": "unknown", "type": "Array"}


def _infer_type_from_default(default_val: Any) -> str:
  """
  Guesses the ODL type string based on a python default value.

  Args:
      default_val: The default value found in the spec.

  Returns:
      A type string ("int", "float", "bool", "Array") or "Any".
  """
  if isinstance(default_val, bool):
    return "bool"
  if isinstance(default_val, int):
    return "int"
  if isinstance(default_val, float):
    return "float"
  if isinstance(default_val, (list, tuple)):
    # Heuristic: verify contents
    if default_val and isinstance(default_val[0], int):
      return "List[int]"
    return "List[Any]"
  return "Any"


def generate_input_value_code(name: str, arg_def: Union[str, Dict[str, Any]]) -> str:
  """
  Generates Python code string to instantiate inputs based on type/constraints.

  Prioritizes:
  1. Explicit `options` list.
  2. Explicit `type` hint.
  3. Inference from `default` value if type is Any.
  4. Inference from `min`/`max` constraints.
  5. Heuristics based on argument name.

  Args:
      name: The argument name (used for heuristics if type is missing).
      arg_def: The normalized argument definition dictionary or type string.

  Returns:
      Valid Python source code expression (e.g., "random.randint(0, 5)").
  """
  # Normalize arg_def if simple string
  if isinstance(arg_def, str):
    arg_def = {"type": arg_def, "name": name}

  arg_type = arg_def.get("type", "Any")
  default_val = arg_def.get("default")

  # 1. Options constraint (Highest Priority)
  if "options" in arg_def and arg_def["options"]:
    opts = arg_def["options"]
    return f"random.choice({opts!r})"

  # 2. Refine "Any" types using Default Value
  if arg_type in [None, "Any"] and default_val is not None:
    arg_type = _infer_type_from_default(default_val)

  # 3. Refine "Any" types using Min/Max constraints
  if arg_type in [None, "Any"]:
    if "min" in arg_def or "max" in arg_def:
      # If explicit bounds provided without type, assume int if bounds are int
      mn = arg_def.get("min")
      mx = arg_def.get("max")
      if isinstance(mn, float) or isinstance(mx, float):
        arg_type = "float"
      else:
        arg_type = "int"

  # 4. Fallback Heuristics for Unknown Types
  if arg_type in [None, "Any"]:
    if name in ["axis", "dim", "keepdim", "keepdims"]:
      return _generate_dim_heuristic(name)
    # Default fallback is Array
    arg_type = "Array"

  # --- Generation Logic ---

  # A. Int Generation
  if arg_type in ["int", "integer"]:
    mn = arg_def.get("min", 1)
    mx = arg_def.get("max", 5)
    # Handle case where user provided only one bound
    if "min" in arg_def and "max" not in arg_def:
      mx = int(mn) + 5
    if "max" in arg_def and "min" not in arg_def:
      mn = int(mx) - 5
    return f"random.randint({mn}, {mx})"

  # B. Bool Generation
  if arg_type in ["bool", "boolean"]:
    return "bool(random.getrandbits(1))"

  # C. Float Generation
  if arg_type in ["float", "number", "double"]:
    mn = arg_def.get("min", 0.0)
    mx = arg_def.get("max", 1.0)
    return f"random.uniform({mn}, {mx})"

  # D. Complex Containers
  if arg_type == "List[int]":
    if default_val is not None and isinstance(default_val, list):
      return repr(default_val)
    return "[1, 2]"
  if "Tuple" in arg_type:
    if default_val is not None and isinstance(default_val, tuple):
      return repr(default_val)
    return "(1, 2)"

  # E. Array/Tensor Generation
  if arg_type in ["Array", "Tensor", "np.ndarray"]:
    return _generate_array_code(arg_def)

  return "None"


def _generate_dim_heuristic(name: str) -> str:
  """
  Helper for dimension argument heuristics strings.

  Args:
      name: Name of the argument (e.g. 'axis').

  Returns:
      Python code string.
  """
  if name in ["axis", "dim"]:
    return "1"
  if name in ["keepdims", "keepdim"]:
    return "bool(random.getrandbits(1))"
  return "1"


def _generate_array_code(arg_def: Dict[str, Any]) -> str:
  """
  Helper for array code generation logic.

  Args:
      arg_def: Argument definition dict with constraints.

  Returns:
      Python code string for numpy logic.
  """
  mn = arg_def.get("min")
  mx = arg_def.get("max")
  dtype = arg_def.get("dtype")

  cast_str = ""
  if dtype:
    cast_str = f".astype(np.{dtype})"
  else:
    cast_str = ".astype(np.float32)"

  if mn is not None and mx is not None:
    return f"np.random.uniform({mn}, {mx}, size=(2, 2)){cast_str}"
  elif mn is not None:
    return f"(np.abs(np.random.randn(2, 2)) + {mn}){cast_str}"

  return f"np.random.randn(2, 2, 2){cast_str}"
