"""Hypothesi Strategies for ODL Types.

This module maps Operation Definition Language (ODL) type strings (e.g., ``Array['N']``,
``List[int]``) into executable Hypothesis search strategies. It handles:

1.  **Primitives**: Constraints-aware generation for ints, floats, bools.
2.  **Tensors**: NumPy array generation with specific dtypes, ranks, and symbolic shapes.
3.  **Containers**: Recursive generation of Lists, Tuples, and Dictionaries.
4.  **Symbolic Consistency**: Ensuring named dimensions (e.g., 'N') resolve consistently
    across different arguments using a shared context.
"""

import logging
import typing

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.testing.fuzzer.generators import NumpyScalar
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


def _get_dtype_strategy(dtype_str: str | None) -> type | np.dtype[NumpyScalar]:
  """Resolve a string dtype representation to a NumPy dtype or type class.

  Args:
      dtype_str: The type string (e.g. 'float32', 'int').

  Returns:
      The corresponding NumPy dtype object or type class.
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
  except (TypeError, ValueError, KeyError) as exc:
    logging.getLogger(__name__).debug("Unrecognized numpy dtype '%s', defaulting to float32: %s", dtype_str, exc)
  return np.float32


def _primitive_strategy(
  type_str: PrimitiveType,
  constraints: typing.Mapping[str, RuntimeTensorOrValue],
) -> st.SearchStrategy[RuntimeTensorOrValue] | None:
  """Build a Hypothesis strategy for recognized scalar/dtype ``PrimitiveType`` hints.

  Args:
      type_str: Parsed ``PrimitiveType`` node.
      constraints: Dictionary of semantic bounds (``min``, ``max``).

  Returns:
      Matching ``SearchStrategy`` if ``type_str`` is a recognized primitive, otherwise ``None``.
  """
  name = type_str.name
  if name in ("int", "integer"):
    raw_mn = constraints.get("min")
    raw_mx = constraints.get("max")
    mn_i = int(raw_mn) if isinstance(raw_mn, (int, float)) else None
    mx_i = int(raw_mx) if isinstance(raw_mx, (int, float)) else None
    return st.integers(min_value=mn_i, max_value=mx_i)

  if name in ("float", "double", "number"):
    raw_mn_f = constraints.get("min", -1e3)
    raw_mx_f = constraints.get("max", 1e3)
    mn_f = float(raw_mn_f) if isinstance(raw_mn_f, (int, float)) else -1e3
    mx_f = float(raw_mx_f) if isinstance(raw_mx_f, (int, float)) else 1e3
    return st.floats(min_value=mn_f, max_value=mx_f, allow_nan=False, allow_infinity=False)

  if name in ("bool", "boolean"):
    return st.booleans()

  if name in ("str", "string"):
    cs_cat: typing.Literal["Cs"] = "Cs"
    return st.text(alphabet=st.characters(blacklist_categories=(cs_cat,)), min_size=1, max_size=10)

  if "dtype" in name.lower():
    return st.sampled_from([np.float32, np.int32, np.float64, np.bool_])

  return None


def _callable_strategy() -> st.SearchStrategy[RuntimeTensorOrValue]:
  """Build a Hypothesis strategy yielding a deterministic identity callable.

  Returns:
      Strategy producing a passthrough callable.
  """

  def _identity_fn(
    x: RuntimeTensorOrValue = None,
    *_args: RuntimeTensorOrValue,
    **_kwargs: RuntimeTensorOrValue,
  ) -> RuntimeTensorOrValue:
    """Return the first input argument unchanged.

    Args:
        x: First input value passed to the generated callable.
        *_args: Additional positional arguments.
        **_kwargs: Additional keyword arguments.

    Returns:
        The unchanged first input value.
    """
    return x

  return st.just(_identity_fn)


def _tuple_strategy(
  type_str: TupleType,
  constraints: typing.Mapping[str, RuntimeTensorOrValue],
  shared_dims: dict[str, st.SearchStrategy[int]] | None,
) -> st.SearchStrategy[RuntimeTensorOrValue]:
  """Build a Hypothesis strategy for fixed or variadic ``TupleType`` nodes.

  Args:
      type_str: Parsed ``TupleType`` node.
      constraints: Semantic constraints dictionary.
      shared_dims: Shared symbolic dimension strategies.

  Returns:
      Strategy producing tuples matching ``type_str``.
  """
  if type_str.variadic:
    inner = type_str.elements[0] if type_str.elements else AnyType()
    return st.lists(strategies_from_spec(inner, constraints, shared_dims), min_size=1, max_size=4).map(tuple)
  sub_strats = [strategies_from_spec(s, constraints, shared_dims) for s in type_str.elements]
  return st.tuples(*sub_strats)


def _dict_strategy(
  type_str: DictType,
  constraints: typing.Mapping[str, RuntimeTensorOrValue],
  shared_dims: dict[str, st.SearchStrategy[int]] | None,
) -> st.SearchStrategy[RuntimeTensorOrValue]:
  """Build a Hypothesis strategy for string-keyed ``DictType`` nodes.

  Args:
      type_str: Parsed ``DictType`` node.
      constraints: Semantic constraints dictionary.
      shared_dims: Shared symbolic dimension strategies.

  Returns:
      Strategy producing dictionaries matching ``type_str``.
  """
  k_ref = type_str.key_type
  key_strat = strategies_from_spec(k_ref, constraints, shared_dims)
  if isinstance(k_ref, (TensorType, ListType, DictType)):
    str_key_strat: st.SearchStrategy[str] = key_strat.map(str)
  else:
    str_key_strat = key_strat.map(lambda k: k if isinstance(k, str) else str(k))
  val_strat = strategies_from_spec(type_str.value_type, constraints, shared_dims)
  return st.dictionaries(keys=str_key_strat, values=val_strat, min_size=1, max_size=3)


def _container_strategy(
  type_str: ParsedType,
  constraints: typing.Mapping[str, RuntimeTensorOrValue],
  shared_dims: dict[str, st.SearchStrategy[int]] | None,
) -> st.SearchStrategy[RuntimeTensorOrValue] | None:
  """Build a Hypothesis strategy for ``NoneType``, ``OptionalType``, ``ListType``, ``TupleType``, or ``DictType``.

  Args:
      type_str: Parsed type AST node.
      constraints: Semantic constraints dictionary.
      shared_dims: Shared symbolic dimension strategies.

  Returns:
      Matching container ``SearchStrategy`` if applicable, otherwise ``None``.
  """
  if isinstance(type_str, NoneType):
    return st.none()
  if isinstance(type_str, OptionalType):
    return st.one_of(st.none(), strategies_from_spec(type_str.inner, constraints, shared_dims))
  if isinstance(type_str, ListType):
    return st.lists(strategies_from_spec(type_str.inner, constraints, shared_dims), min_size=1, max_size=4)
  if isinstance(type_str, TupleType):
    return _tuple_strategy(type_str, constraints, shared_dims)
  if isinstance(type_str, DictType):
    return _dict_strategy(type_str, constraints, shared_dims)
  return None


def strategies_from_spec(
  type_str: str | ParsedType,
  constraints: typing.Mapping[str, RuntimeTensorOrValue] | None,
  shared_dims: dict[str, st.SearchStrategy[int]] | None = None,
) -> st.SearchStrategy[RuntimeTensorOrValue]:
  """Construct a Hypothesis strategy from a type string and constraints.

  Recursively parses complex types (e.g., ``List[int]``) and delegates
  array creation to `_array_strategy`.

  Args:
      type_str: The ODL type hint (e.g. "int", "Array['N']").
      constraints: Dictionary of constraints (min, max, options).
      shared_dims: Mutable dictionary mapping symbol names to shared integer strategies.

  Returns:
      A valid Hypothesis SearchStrategy.

  """
  constrs = constraints or {}
  raw_opts = constrs.get("options")
  if isinstance(raw_opts, (list, tuple)) and raw_opts:
    return st.sampled_from(raw_opts)

  parsed = parse_type_annotation(type_str) if isinstance(type_str, str) else type_str

  if isinstance(parsed, UnionType):
    return st.one_of(*[strategies_from_spec(p, constrs, shared_dims) for p in parsed.types])
  if isinstance(parsed, TensorType):
    return _array_strategy(parsed, constrs, shared_dims)
  if isinstance(parsed, PrimitiveType):
    prim_strat = _primitive_strategy(parsed, constrs)
    if prim_strat is not None:
      return prim_strat
  if isinstance(parsed, CallableType):
    return _callable_strategy()

  cont_strat = _container_strategy(parsed, constrs, shared_dims)
  if cont_strat is not None:
    return cont_strat

  if "default" in constrs:
    return st.just(constrs["default"])
  return _array_strategy(TensorType(dims=None), constrs, shared_dims)


def _array_strategy(
  type_str: "TensorType | ParsedType",
  constraints: typing.Mapping[str, RuntimeTensorOrValue],
  shared_dims: dict[str, st.SearchStrategy[int]] | None,
) -> st.SearchStrategy[np.ndarray[tuple[int, ...], np.dtype[NumpyScalar]]]:
  """Construct a NumPy array strategy based on rank, symbolic shape, and element constraints.

  Args:
      type_str: Parsed TensorType.
      constraints: User defined constraints (dtype, min, max).
      shared_dims: Dictionary for resolving shared symbolic dimensions.

  Returns:
      A strategy generating np.ndarray.

  """
  raw_dt = constraints.get("dtype")
  dtype = _get_dtype_strategy(str(raw_dt) if isinstance(raw_dt, str) else None)

  dims: list[st.SearchStrategy[int]] | None = None

  raw_dims = type_str.dims if isinstance(type_str, TensorType) else None
  if raw_dims:
    dims = []
    for d in raw_dims:
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
  else:
    raw_rank = constraints.get("rank")
    if isinstance(raw_rank, (int, float)):
      dims = [st.integers(min_value=1, max_value=8) for _ in range(int(raw_rank))]
    else:
      # IMPORTANT: Set min_side=1 to prevent generating (0,) shapes which trivially pass validation
      # ("Empty vs Empty" passes, hiding logic bugs).
      return npst.arrays(dtype, shape=npst.array_shapes(min_dims=1, max_dims=4, min_side=1))

  # Construct the concrete shape strategy
  shape_strat = st.tuples(*dims)

  raw_mn = constraints.get("min")
  raw_mx = constraints.get("max")

  from hypothesis.strategies import SearchStrategy

  elements: SearchStrategy[int | float] | None = None
  if np.issubdtype(dtype, np.integer):
    min_v_i = int(raw_mn) if isinstance(raw_mn, (int, float)) else -10
    max_v_i = int(raw_mx) if isinstance(raw_mx, (int, float)) else 10
    elements = st.integers(min_value=min_v_i, max_value=max_v_i)

  elif np.issubdtype(dtype, np.floating):
    min_v_f = float(raw_mn) if isinstance(raw_mn, (int, float)) else -10.0
    max_v_f = float(raw_mx) if isinstance(raw_mx, (int, float)) else 10.0
    elements = st.floats(
      min_value=min_v_f,
      max_value=max_v_f,
      allow_nan=False,
      allow_infinity=False,
    )

  return npst.arrays(dtype, shape=shape_strat, elements=elements)
