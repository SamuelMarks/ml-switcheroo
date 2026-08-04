"""Type Hint Parser and Recursive Generation Logic.

This module processes string type hints (e.g. `List[Array['N']]`) and
generates conforming data structures for runtime fuzzing.

It includes logic to:
1.  Parse nested generic types.
2.  Resolve symbolic shape constraints.
3.  Respect semantic constraints (`min`, `max`, `options`).
4.  **Infer generation strategy from default values** when explicit hints are erased.
"""

import random
import numpy as np
from typing import Any, Dict, Tuple, Optional, Union

from ml_switcheroo.testing.fuzzer.type_parser import (
  parse_type_annotation,
  ParsedType,
  AnyType,
  NoneType,
  PrimitiveType,
  UnionType,
  OptionalType,
  TupleType,
  ListType,
  DictType,
  TensorType,
  CallableType,
)


from ml_switcheroo.testing.fuzzer.generators import (
  generate_array,
  generate_scalar_int,
  generate_scalar_float,
  generate_fake_callable,
)
from ml_switcheroo.testing.fuzzer.utils import (
  resolve_symbolic_shape,
  adjust_shape_rank,
)


def get_fallback_base_value(parsed: ParsedType, base_shape: Tuple[int, ...]) -> Any:
  """Returns a minimal valid value to terminate recursion when depth limit is reached.

  Args:
      parsed: The parsed type hint.
      base_shape: Default shape for array fallbacks.

  Returns:
      Safe fallback value (0, empty list, etc).

  """
  if isinstance(parsed, PrimitiveType):
    if parsed.name in ["bool", "boolean"]:
      return False
    if parsed.name in ["int", "integer"]:
      return 0
    if parsed.name in ["float", "double", "number"]:
      return 0.0
    if parsed.name in ["str", "string"]:
      return ""
    if "dtype" in parsed.name.lower():
      return np.float32

  if isinstance(parsed, TensorType):
    return np.zeros(base_shape, dtype=np.float32)
  if isinstance(parsed, ListType):
    return []
  if isinstance(parsed, TupleType):
    return ()
  if isinstance(parsed, DictType):
    return {}
  if isinstance(parsed, CallableType):
    return generate_fake_callable()

  return None


def generate_from_hint(
  type_hint: Union[str, ParsedType],
  base_shape: Tuple[int, ...],
  depth: int,
  max_depth: int,
  symbol_map: Dict[str, int],
  constraints: Optional[Dict[str, Any]] = None,
) -> Any:
  """Recursively parses a type hint and generates conforming data.

  If type hints are generic ("Any"), it attempts to infer the type logic
  from the `default` value provided in constraints.

  Args:
      type_hint: The type hint string or ParsedType to parse.
      base_shape: Default shape helper.
      depth: Current recursion depth.
      max_depth: Limit for recursion.
      symbol_map: Shared context for dimension symbols ('N').
      constraints: Semantic constraints (min, max, options, default).

  Returns:
      Generated data structure.

  """
  if isinstance(type_hint, str):
    type_hint = parse_type_annotation(type_hint)

  if depth > max_depth:
    return get_fallback_base_value(type_hint, base_shape)

  constrs = constraints or {}

  # 0. Options Override: if explicit options provided, pick one
  if "options" in constrs and constrs["options"]:
    return random.choice(constrs["options"])

  # 1. Inference from Default Value (if Type is Any/Unknown)
  if isinstance(type_hint, AnyType) and "default" in constrs:
    default_val = constrs["default"]
    # Probabilistic: Sometimes just use the default value directly (Coverage)
    if random.random() < 0.2:
      return default_val

    # Otherwise infer type to generate VARIATIONS of that type
    if isinstance(default_val, bool):
      type_hint = PrimitiveType(name="bool")
    elif isinstance(default_val, int):
      type_hint = PrimitiveType(name="int")
    elif isinstance(default_val, float):
      type_hint = PrimitiveType(name="float")
    elif isinstance(default_val, list):
      if default_val and isinstance(default_val[0], int):
        type_hint = ListType(inner=PrimitiveType(name="int"))
      else:
        type_hint = ListType(inner=AnyType())

  # 2. Unions
  if isinstance(type_hint, UnionType):
    chosen = random.choice(type_hint.types)
    return generate_from_hint(chosen, base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 3. Optional
  if isinstance(type_hint, OptionalType):
    if random.random() < 0.2:
      return None
    return generate_from_hint(type_hint.inner, base_shape, depth + 1, max_depth, symbol_map, constraints)

  # 4. Tuple
  if isinstance(type_hint, TupleType):
    if type_hint.variadic:
      elem_type = type_hint.elements[0] if type_hint.elements else AnyType()
      length = random.randint(1, 3)
      return tuple(
        generate_from_hint(elem_type, base_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length)
      )
    else:
      return tuple(
        generate_from_hint(t, base_shape, depth + 1, max_depth, symbol_map, constraints) for t in type_hint.elements
      )

  # 5. List/Sequence
  if isinstance(type_hint, ListType):
    inner = type_hint.inner
    length = random.randint(2, 3)

    is_tensor = isinstance(inner, TensorType)
    first_elem = generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints)

    if is_tensor and isinstance(first_elem, np.ndarray):
      uniform_shape = first_elem.shape
      list_data = [first_elem]
      for _ in range(length - 1):
        elem = generate_from_hint(inner, uniform_shape, depth + 1, max_depth, symbol_map, constraints)
        if isinstance(elem, np.ndarray) and elem.shape != uniform_shape:
          elem = generate_array("float", uniform_shape, constrs)
        list_data.append(elem)
      return list_data
    else:
      list_data = [first_elem]
      for _ in range(length - 1):
        list_data.append(generate_from_hint(inner, base_shape, depth + 1, max_depth, symbol_map, constraints))
      return list_data

  # 6. Dict/Mapping
  if isinstance(type_hint, DictType):
    if isinstance(type_hint.key_type, AnyType) and isinstance(type_hint.value_type, AnyType):
      return {}
    length = random.randint(1, 3)
    data = {}
    for _ in range(length):
      k = generate_from_hint(type_hint.key_type, base_shape, depth + 1, max_depth, symbol_map)
      v = generate_from_hint(type_hint.value_type, base_shape, depth + 1, max_depth, symbol_map, constraints)
      if isinstance(k, (list, dict, np.ndarray, np.generic)):
        k = str(k)
      data[k] = v
    return data

  # 7. Nulls
  if isinstance(type_hint, NoneType):
    return None

  # 8. Tensors
  if isinstance(type_hint, TensorType):
    if type_hint.dims:
      dims_str = ",".join(type_hint.dims)
      shape = resolve_symbolic_shape(dims_str, symbol_map)
    else:
      shape = base_shape

    if constrs.get("rank"):
      shape = adjust_shape_rank(shape, constrs["rank"])
    return generate_array("float", shape, constrs)

  # 9. Callables
  if isinstance(type_hint, CallableType):
    return generate_fake_callable(constrs)

  # 10. Primitives
  if isinstance(type_hint, PrimitiveType):
    name = type_hint.name
    if name in ["int", "integer"]:
      return generate_scalar_int(constrs)
    if name in ["float", "double", "number"]:
      return generate_scalar_float(constrs)
    if name in ["bool", "boolean"]:
      return bool(random.getrandbits(1))
    if name in ["str", "string"]:
      return "val_" + str(random.randint(0, 100))
    if "dtype" in name.lower():
      return random.choice([np.float32, np.int32, np.float64, np.bool_])

  # Fallback for unknown strings using default heuristics for arrays
  return generate_array("float", base_shape, constrs)
