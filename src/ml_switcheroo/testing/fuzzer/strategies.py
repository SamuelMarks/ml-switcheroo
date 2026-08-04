"""Hypothesis Strategies for ODL Types.

This module maps Operation Definition Language (ODL) type strings (e.g., ``Array['N']``,
``List[int]``) into executable Hypothesis search strategies. It handles:

1.  **Primitives**: Constraints-aware generation for ints, floats, bools.
2.  **Tensors**: Numpy array generation with specific dtypes, ranks, and symbolic shapes.
3.  **Containers**: Recursive generation of Lists, Tuples, and Dictionaries.
4.  **Symbolic Consistency**: Ensuring named dimensions (e.g., 'N') resolve consistently
    across different arguments using a shared context.
"""

from typing import Any

import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from typing import Dict, Optional, Union

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


def _get_dtype_strategy(dtype_str: Optional[str]) -> Any:
  """Resolves a string dtype representation to a Numpy dtype or type class.

  Args:
      dtype_str: The type string (e.g. 'float32', 'int').

  Returns:
      The corresponding numpy dtype object or type class.
      Defaults to ``np.float32`` if unknown or None.

  """
  if not dtype_str:
    return np.float32
  if dtype_str in ["int", "int32"]:
    return np.int32
  if dtype_str in ["int64", "long"]:
    return np.int64
  if dtype_str in ["float", "float32"]:
    return np.float32
  if dtype_str in ["float64", "double"]:
    return np.float64
  if dtype_str in ["bool"]:
    return bool
  try:
    return np.dtype(dtype_str)
  except Exception:
    pass
  return np.float32


def strategies_from_spec(
  type_str: Union[str, ParsedType],
  constraints: Dict[str, Any],
  shared_dims: Optional[Dict[str, Any]] = None,
) -> st.SearchStrategy:
  """Constructs a Hypothesis strategy from a type string and constraints.

  Recursively parses complex types (e.g., ``List[int]``) and delegates
  array creation to `_array_strategy`.

  Args:
      type_str: The ODL type hint (e.g. "int", "Array['N']").
      constraints: Dictionary of constraints (min, max, options).
      shared_dims: Mutable dictionary mapping symbol names to shared integer strategies.

  Returns:
      A valid Hypothesis SearchStrategy.

  """
  constraints = constraints or {}

  if "options" in constraints and constraints["options"]:
    return st.sampled_from(constraints["options"])

  if isinstance(type_str, str):
    type_str = parse_type_annotation(type_str)

  # 0. Union Types
  if isinstance(type_str, UnionType):
    return st.one_of(*[strategies_from_spec(p, constraints, shared_dims) for p in type_str.types])

  # 1. Arrays / Tensors
  if isinstance(type_str, TensorType):
    return _array_strategy(type_str, constraints, shared_dims)

  # 2. Primitives
  if isinstance(type_str, PrimitiveType):
    name = type_str.name
    if name in ("int", "integer"):
      mn = constraints.get("min")
      mx = constraints.get("max")
      return st.integers(min_value=mn, max_value=mx)

    if name in ("float", "double", "number"):
      mn = float(constraints.get("min", -1e3))
      mx = float(constraints.get("max", 1e3))
      return st.floats(min_value=mn, max_value=mx, allow_nan=False, allow_infinity=False)

    if name in ("bool", "boolean"):
      return st.booleans()

    if name in ("str", "string"):
      return st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=10)

    if "dtype" in name.lower():
      return st.sampled_from([np.float32, np.int32, np.float64, np.bool_])

  if isinstance(type_str, CallableType):
    return st.just(lambda x, *args, **kwargs: x)

  # 3. Containers
  if isinstance(type_str, NoneType):
    return st.none()

  if isinstance(type_str, OptionalType):
    return st.one_of(st.none(), strategies_from_spec(type_str.inner, constraints, shared_dims))

  if isinstance(type_str, ListType):
    return st.lists(strategies_from_spec(type_str.inner, constraints, shared_dims), min_size=1, max_size=4)

  if isinstance(type_str, TupleType):
    if type_str.variadic:
      inner = type_str.elements[0] if type_str.elements else AnyType()
      return st.lists(strategies_from_spec(inner, constraints, shared_dims), min_size=1, max_size=4).map(tuple)
    else:
      sub_strats = [strategies_from_spec(s, constraints, shared_dims) for s in type_str.elements]
      return st.tuples(*sub_strats)

  if isinstance(type_str, DictType):
    k_ref = type_str.key_type
    v_ref = type_str.value_type

    key_strat = strategies_from_spec(k_ref, constraints, shared_dims)
    if isinstance(k_ref, (TensorType, ListType, DictType)):
      key_strat = key_strat.map(str)

    val_strat = strategies_from_spec(v_ref, constraints, shared_dims)

    return st.dictionaries(
      keys=key_strat,
      values=val_strat,
      min_size=1,
      max_size=3,
    )

  # Inference fallback
  if "default" in constraints:
    return st.just(constraints["default"])

  # Fallback default
  return _array_strategy(TensorType(dims=None), constraints, shared_dims)


def _array_strategy(
  type_str: TensorType, constraints: Dict[Any, Any], shared_dims: Optional[Dict[Any, Any]]
) -> st.SearchStrategy:
  """Constructs a numpy array strategy based on rank, symbolic shape, and element constraints.

  Args:
      type_str: Parsed TensorType.
      constraints: User defined constraints (dtype, min, max).
      shared_dims: Dictionary for resolving shared symbolic dimensions.

  Returns:
      A strategy generating np.ndarray.

  """
  dtype = _get_dtype_strategy(constraints.get("dtype"))

  dims = None

  if type_str.dims:
    dims = []
    for d in type_str.dims:
      d = d.strip().replace("'", "").replace('"', "")
      if d.isdigit():
        # Fixed dimension
        dims.append(st.just(int(d)))
      elif d.isidentifier() and shared_dims is not None:
        # Symbolic dimension
        if d not in shared_dims:
          # Define symbol (1 to 8 size) in shared scope
          shared_dims[d] = st.shared(st.integers(min_value=1, max_value=8), key=d)
        dims.append(shared_dims[d])
      else:
        # Unbound dimension (random)
        dims.append(st.integers(min_value=1, max_value=8))

  if dims is None:
    rank = constraints.get("rank")
    if rank is not None:
      dims = [st.integers(min_value=1, max_value=8) for _ in range(rank)]
    else:
      # IMPORTANT: Set min_side=1 to prevent generating (0,) shapes which trivially pass validation
      # ("Empty vs Empty" passes, hiding logic bugs).
      return npst.arrays(dtype, shape=npst.array_shapes(min_dims=1, max_dims=4, min_side=1))

  # Construct the concrete shape strategy
  shape_strat = st.tuples(*dims)

  mn = constraints.get("min")
  mx = constraints.get("max")

  elements = None
  if np.issubdtype(dtype, np.integer):
    min_v = int(mn) if mn is not None else -10
    max_v = int(mx) if mx is not None else 10
    elements = st.integers(min_value=min_v, max_value=max_v)

  elif np.issubdtype(dtype, np.floating):
    min_v = float(mn) if mn is not None else -10.0  # type: ignore
    max_v = float(mx) if mx is not None else 10.0  # type: ignore
    elements = st.floats(  # type: ignore
      min_value=min_v,
      max_value=max_v,
      allow_nan=False,
      allow_infinity=False,
    )

  return npst.arrays(dtype, shape=shape_strat, elements=elements)
