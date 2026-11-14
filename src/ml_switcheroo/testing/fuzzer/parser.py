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
import typing

import numpy as np

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.testing.fuzzer.generators import (
  generate_array,
  generate_scalar_float,
  generate_scalar_int,
  generate_symbolic_callable,
)
from ml_switcheroo.testing.fuzzer.type_parser import (
  AnyType,
  CallableType,
  DictType,
  ListType,
  NoneType,
  OptionalType,
  ParsedType,
  PrimitiveType,
  TensorType,
  TupleType,
  UnionType,
  parse_type_annotation,
)
from ml_switcheroo.testing.fuzzer.utils import adjust_shape_rank, resolve_symbolic_shape


def get_fallback_base_value(parsed: "ParsedType", base_shape: tuple[int, ...]) -> RuntimeTensorOrValue:
  """Return a minimal valid value to terminate recursion when depth limit is reached.

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
    return generate_symbolic_callable()

  return None


def _normalize_base_shape(
  base_shape: tuple[int, ...] | typing.Sequence[int] | RuntimeTensorOrValue,
) -> tuple[int, ...]:
  """Normalize ``base_shape`` into an explicit integer dimension tuple.

  Args:
      base_shape: Candidate tuple, list, or fallback shape descriptor.

  Returns:
      Normalized integer shape tuple, defaulting to ``(2, 2)``.
  """
  if isinstance(base_shape, (tuple, list)):
    return tuple(int(x) for x in base_shape if isinstance(x, (int, float, str)))
  return (2, 2)


def _infer_type_from_default_value(default_val: RuntimeTensorOrValue) -> ParsedType:
  """Infer a concrete ``ParsedType`` from a constraint ``default`` value.

  Args:
      default_val: Default parameter value from semantic constraints.

  Returns:
      Inferred ``ParsedType`` node, or ``AnyType`` when structure is opaque.
  """
  if isinstance(default_val, bool):
    return PrimitiveType(name="bool")
  if isinstance(default_val, int):
    return PrimitiveType(name="int")
  if isinstance(default_val, float):
    return PrimitiveType(name="float")
  if isinstance(default_val, list):
    if default_val and isinstance(default_val[0], int):
      return ListType(inner=PrimitiveType(name="int"))
    return ListType(inner=AnyType())
  return AnyType()


def _generate_tuple_value(
  type_hint: TupleType,
  norm_shape: tuple[int, ...],
  depth: int,
  max_depth: int,
  symbol_map: dict[str, int],
  constraints: typing.Mapping[str, RuntimeTensorOrValue] | None,
) -> tuple[RuntimeTensorOrValue, ...]:
  """Generate a fixed or variadic tuple matching ``type_hint``.

  Args:
      type_hint: Parsed ``TupleType`` node.
      norm_shape: Normalized fallback tensor shape.
      depth: Current recursion depth.
      max_depth: Maximum recursion depth.
      symbol_map: Shared symbolic dimension bindings.
      constraints: Optional semantic constraints.

  Returns:
      Generated tuple of conforming element values.
  """
  if type_hint.variadic:
    elem_type = type_hint.elements[0] if type_hint.elements else AnyType()
    length = random.randint(1, 3)
    return tuple(
      generate_from_hint(elem_type, norm_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length)
    )
  return tuple(
    generate_from_hint(t, norm_shape, depth + 1, max_depth, symbol_map, constraints) for t in type_hint.elements
  )


def _generate_list_value(
  type_hint: ListType,
  norm_shape: tuple[int, ...],
  depth: int,
  max_depth: int,
  symbol_map: dict[str, int],
  constraints: typing.Mapping[str, RuntimeTensorOrValue] | None,
  constrs: typing.Mapping[str, RuntimeTensorOrValue],
) -> list[RuntimeTensorOrValue]:
  """Generate a list of values matching ``type_hint`` with uniform tensor shapes when required.

  Args:
      type_hint: Parsed ``ListType`` node.
      norm_shape: Normalized fallback tensor shape.
      depth: Current recursion depth.
      max_depth: Maximum recursion depth.
      symbol_map: Shared symbolic dimension bindings.
      constraints: Raw constraints mapping forwarded recursively.
      constrs: Non-null constraints dictionary for array fallback generation.

  Returns:
      Generated list of conforming element values.
  """
  inner = type_hint.inner
  length = random.randint(2, 3)
  first_elem = generate_from_hint(inner, norm_shape, depth + 1, max_depth, symbol_map, constraints)

  if isinstance(inner, TensorType) and isinstance(first_elem, np.ndarray):
    uniform_shape = first_elem.shape
    list_data: list[RuntimeTensorOrValue] = [first_elem]
    for _ in range(length - 1):
      elem = generate_from_hint(inner, uniform_shape, depth + 1, max_depth, symbol_map, constraints)
      if isinstance(elem, np.ndarray) and elem.shape != uniform_shape:
        elem = generate_array("float", uniform_shape, constrs)
      list_data.append(elem)
    return list_data

  return [first_elem] + [
    generate_from_hint(inner, norm_shape, depth + 1, max_depth, symbol_map, constraints) for _ in range(length - 1)
  ]


def _generate_dict_value(
  type_hint: DictType,
  norm_shape: tuple[int, ...],
  depth: int,
  max_depth: int,
  symbol_map: dict[str, int],
  constraints: typing.Mapping[str, RuntimeTensorOrValue] | None,
) -> dict[str, RuntimeTensorOrValue]:
  """Generate a dictionary matching ``type_hint``.

  Args:
      type_hint: Parsed ``DictType`` node.
      norm_shape: Normalized fallback tensor shape.
      depth: Current recursion depth.
      max_depth: Maximum recursion depth.
      symbol_map: Shared symbolic dimension bindings.
      constraints: Optional semantic constraints for dictionary values.

  Returns:
      Generated string-keyed dictionary.
  """
  if isinstance(type_hint.key_type, AnyType) and isinstance(type_hint.value_type, AnyType):
    return {}
  length = random.randint(1, 3)
  data: dict[str, RuntimeTensorOrValue] = {}
  for _ in range(length):
    k = generate_from_hint(type_hint.key_type, norm_shape, depth + 1, max_depth, symbol_map)
    v = generate_from_hint(type_hint.value_type, norm_shape, depth + 1, max_depth, symbol_map, constraints)
    key_str = k if isinstance(k, str) else str(k)
    data[key_str] = v
  return data


def _generate_tensor_value(
  type_hint: TensorType,
  norm_shape: tuple[int, ...],
  symbol_map: dict[str, int],
  constrs: typing.Mapping[str, RuntimeTensorOrValue],
) -> RuntimeTensorOrValue:
  """Generate a NumPy array matching symbolic dimensions and rank constraints.

  Args:
      type_hint: Parsed ``TensorType`` node.
      norm_shape: Normalized fallback tensor shape.
      symbol_map: Shared symbolic dimension bindings.
      constrs: Semantic constraints dictionary.

  Returns:
      Generated NumPy array.
  """
  shape = resolve_symbolic_shape(",".join(type_hint.dims), symbol_map) if type_hint.dims else norm_shape
  raw_rank = constrs.get("rank")
  if isinstance(raw_rank, (int, float, str)):
    shape = adjust_shape_rank(shape, int(raw_rank))
  return generate_array("float", shape, constrs)


def _generate_primitive_value(
  type_hint: PrimitiveType,
  norm_shape: tuple[int, ...],
  constrs: typing.Mapping[str, RuntimeTensorOrValue],
) -> RuntimeTensorOrValue:
  """Generate a scalar, string, dtype, or array fallback value for ``PrimitiveType``.

  Args:
      type_hint: Parsed ``PrimitiveType`` node.
      norm_shape: Normalized fallback tensor shape for unrecognized primitive names.
      constrs: Semantic constraints dictionary.

  Returns:
      Generated primitive value or fallback array.
  """
  name = type_hint.name
  if name in ("int", "integer"):
    return generate_scalar_int(constrs)
  if name in ("float", "double", "number"):
    return generate_scalar_float(constrs)
  if name in ("bool", "boolean"):
    return bool(random.getrandbits(1))
  if name in ("str", "string"):
    return "val_" + str(random.randint(0, 100))
  if "dtype" in name.lower():
    return random.choice([np.float32, np.int32, np.float64, np.bool_])
  return generate_array("float", norm_shape, constrs)


def generate_from_hint(
  type_hint: str | ParsedType,
  base_shape: tuple[int, ...] | typing.Sequence[int] | RuntimeTensorOrValue,
  depth: int,
  max_depth: int,
  symbol_map: dict[str, int],
  constraints: typing.Mapping[str, RuntimeTensorOrValue] | None = None,
) -> RuntimeTensorOrValue:
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
      Generated value matching the type structure and constraints.

  """
  norm_shape = _normalize_base_shape(base_shape)
  parsed_hint = parse_type_annotation(type_hint) if isinstance(type_hint, str) else type_hint

  if depth > max_depth:
    return get_fallback_base_value(parsed_hint, norm_shape)

  constrs = constraints or {}
  raw_opts = constrs.get("options")
  if isinstance(raw_opts, (list, tuple)) and raw_opts:
    opt_val: RuntimeTensorOrValue = random.choice(raw_opts)
    return opt_val

  if isinstance(parsed_hint, AnyType) and "default" in constrs:
    default_val = constrs["default"]
    if random.random() < 0.2:
      return default_val
    parsed_hint = _infer_type_from_default_value(default_val)

  if isinstance(parsed_hint, UnionType):
    return generate_from_hint(random.choice(parsed_hint.types), norm_shape, depth + 1, max_depth, symbol_map, constraints)
  if isinstance(parsed_hint, OptionalType):
    if random.random() < 0.2:
      return None
    return generate_from_hint(parsed_hint.inner, norm_shape, depth + 1, max_depth, symbol_map, constraints)
  if isinstance(parsed_hint, TupleType):
    return _generate_tuple_value(parsed_hint, norm_shape, depth, max_depth, symbol_map, constraints)
  if isinstance(parsed_hint, ListType):
    return _generate_list_value(parsed_hint, norm_shape, depth, max_depth, symbol_map, constraints, constrs)
  if isinstance(parsed_hint, DictType):
    return _generate_dict_value(parsed_hint, norm_shape, depth, max_depth, symbol_map, constraints)
  if isinstance(parsed_hint, NoneType):
    return None
  if isinstance(parsed_hint, TensorType):
    return _generate_tensor_value(parsed_hint, norm_shape, symbol_map, constrs)
  if isinstance(parsed_hint, CallableType):
    return generate_symbolic_callable(constrs)
  if isinstance(parsed_hint, PrimitiveType):
    return _generate_primitive_value(parsed_hint, norm_shape, constrs)
  return generate_array("float", norm_shape, constrs)
