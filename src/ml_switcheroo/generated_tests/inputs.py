"""
Input Value Code Generation.

This module handles the logic for generating Python *source code strings* that
instantiate test data. It parses argument definitions and emits code like
`random.randint(1, 5)` or `np.random.randn(...)`.
"""

from typing import Union, Dict, Any


def parse_arg_def(arg: Union[str, tuple, dict]) -> Dict[str, Any]:
  """
  Normalizes a heterogeneous argument definition into a standard dictionary.

  Args:
      arg: An argument definition. Can be a string ("x"), a tuple ("x", "int"),
           or a dictionary (ODL schema).

  Returns:
      Dict: Normalized dict with keys 'name', 'type', and optional constraints.
  """
  if isinstance(arg, str):
    return {"name": arg, "type": "Array"}
  elif isinstance(arg, (tuple, list)) and len(arg) == 2:
    return {"name": arg[0], "type": arg[1]}
  elif isinstance(arg, dict):
    # Shallow copy to avoid mutation
    arg = arg.copy()
    if "type" not in arg:
      # Infer basic type from constraints if explicit type missing
      if isinstance(arg.get("min"), int) or isinstance(arg.get("max"), int):
        arg["type"] = "int"
      elif isinstance(arg.get("min"), float):
        arg["type"] = "float"
      else:
        arg["type"] = "Array"
    return arg
  return {"name": "unknown", "type": "Array"}


def generate_input_value_code(name: str, arg_def: Union[str, Dict[str, Any]]) -> str:
  """
  Generates Python code string to instantiate inputs based on type/constraints.

  Args:
      name: The argument name (used for heuristics if type is missing).
      arg_def: The normalized argument definition dictionary or type string.

  Returns:
      str: Valid Python source code expression (e.g., "random.randint(0, 5)").
  """
  # Normalize arg_def if simple string
  if isinstance(arg_def, str):
    arg_def = {"type": arg_def}

  arg_type = arg_def.get("type")

  # Heuristic for unknown types
  if arg_type in [None, "Any"]:
    if name in ["axis", "dim"]:
      return "1"
    if name == "keepdims":
      return "bool(random.getrandbits(1))"
    # Default fallback for Any/None -> Array logic or None
    arg_type = "Array"

  # 1. Options constraint
  if "options" in arg_def:
    opts = arg_def["options"]
    return f"random.choice({opts!r})"

  # 2. Int Range / Int Type
  if arg_type == "int":
    mn = arg_def.get("min", 1)
    mx = arg_def.get("max", 5)
    return f"random.randint({mn}, {mx})"

  # 3. Bool
  if arg_type == "bool":
    return "bool(random.getrandbits(1))"

  # 4. Float Range
  if arg_type == "float":
    mn = arg_def.get("min", 0.0)
    mx = arg_def.get("max", 1.0)
    return f"random.uniform({mn}, {mx})"

  # 5. Complex Types
  if arg_type == "List[int]":
    return "[1, 2]"
  if "Tuple" in arg_type:
    return "(1, 2)"

  # 6. Array/Tensor constraints
  if arg_type in ["Array", "Tensor"]:
    mn = arg_def.get("min")
    mx = arg_def.get("max")

    if mn is not None and mx is not None:
      return f"np.random.uniform({mn}, {mx}, size=(2, 2)).astype(np.float32)"
    elif mn is not None:
      return f"np.abs(np.random.randn(2, 2).astype(np.float32)) + {mn}"

    return "np.random.randn(2, 2, 2).astype(np.float32)"

  return "None"
