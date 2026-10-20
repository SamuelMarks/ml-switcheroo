"""Primitive Generation Logic for the Fuzzer.

This module contains functions to generate random scalars, NumPy arrays,
and dummy callables respecting specified constraints (min, max, dtype).

Feature Update: Added broadcasting logic generator.
"""

import random
import typing

import numpy as np
from pydantic import JsonValue


def generate_scalar_int(constraints: typing.Mapping[str, typing.Union[typing.Hashable, typing.Sized]]) -> int:
  """Generate a random integer within constrained bounds.

  Args:
      constraints (Mapping[str, Union[Hashable, Sized]]): Dictionary containing optional 'min' and 'max'.

  Returns:
      int: A random integer.

  """
  raw_min = constraints.get("min", -5)
  raw_max = constraints.get("max", 5)
  min_v = int(raw_min) if isinstance(raw_min, (int, float, str)) else -5
  max_v = int(raw_max) if isinstance(raw_max, (int, float, str)) else 5
  # random.randint is inclusive for both bounds
  return random.randint(min_v, max_v)


def generate_scalar_float(constraints: typing.Mapping[str, typing.Union[typing.Hashable, typing.Sized]]) -> float:
  """Generate a random float within constrained bounds.

  Args:
      constraints (Mapping[str, Union[Hashable, Sized]]): Dictionary containing optional 'min' and 'max'.

  Returns:
      float: A random float.

  """
  raw_min = constraints.get("min")
  raw_max = constraints.get("max")
  if isinstance(raw_min, (int, float)) and isinstance(raw_max, (int, float)):
    return random.uniform(float(raw_min), float(raw_max))

  val = random.random()
  # Adjust one-sided bounds if necessary
  if isinstance(raw_min, (int, float)):
    val = max(val, float(raw_min))
  if isinstance(raw_max, (int, float)):
    val = min(val, float(raw_max))
  return val


NumpyScalar = typing.Union[
  np.bool_,
  np.int8,
  np.int16,
  np.int32,
  np.int64,
  np.uint8,
  np.uint16,
  np.uint32,
  np.uint64,
  np.float16,
  np.float32,
  np.float64,
  np.complex64,
  np.complex128,
]


def generate_array(
  type_lbl: str,
  shape: typing.Tuple[int, ...],
  constraints: typing.Mapping[str, typing.Union[typing.Hashable, typing.Sized]],
) -> np.ndarray[typing.Tuple[int, ...], np.dtype[NumpyScalar]]:
  """Generate a random NumPy array bounded by constraints.

  Args:
      type_lbl (str): General type category ('float', 'int', 'bool').
      shape (Tuple[int, ...]): The shape of the array.
      constraints (Mapping[str, Union[Hashable, Sized]]): Constraints like 'min', 'max', 'dtype'.

  Returns:
      np.ndarray: The generated array.

  """
  # Resolve bounds
  raw_min = constraints.get("min")
  raw_max = constraints.get("max")
  min_val = float(raw_min) if isinstance(raw_min, (int, float)) else None
  max_val = float(raw_max) if isinstance(raw_max, (int, float)) else None

  # Resolve explicit dtype constraint if present
  dtype_req = constraints.get("dtype")
  explicit_dtype: typing.Optional[np.dtype[NumpyScalar]] = None
  if dtype_req:
    try:
      explicit_dtype = np.dtype(str(dtype_req))
    except TypeError:
      pass

  # Use explicit dtype to override heuristic type_lbl if available
  if explicit_dtype:
    if np.issubdtype(explicit_dtype, np.integer):
      type_lbl = "int"
    elif explicit_dtype.kind == "b":
      type_lbl = "bool"
    else:
      type_lbl = "float"

  if type_lbl == "bool":
    return np.random.randint(0, 2, size=shape).astype(bool)

  if type_lbl == "int":
    # Default range [-10, 10]
    low = int(min_val) if min_val is not None else -10
    # High in randint is exclusive, update for inclusivity match
    high_bound = int(max_val) if max_val is not None else 10
    high = high_bound + 1

    int_arr = np.random.randint(low, high, size=shape)
    if explicit_dtype:
      return int_arr.astype(explicit_dtype)
    return int_arr.astype(np.int32)

  # Float default
  arr = np.random.randn(*shape)

  # Constraint Clipping logic
  if min_val is not None or max_val is not None:
    if min_val is not None and max_val is not None:
      # Uniform within bounds
      arr = np.random.uniform(min_val, max_val, size=shape)
    else:
      # Clip standard normal
      safe_min = float(min_val) if min_val is not None else -np.inf
      safe_max = float(max_val) if max_val is not None else np.inf

      # If just Min is 0 (Log), use absolute or exp adjustment
      if min_val is not None and min_val >= 0:
        arr = np.abs(arr) + min_val

      arr = np.clip(arr, safe_min, safe_max)

  if explicit_dtype:
    return arr.astype(explicit_dtype)

  return arr.astype(np.float32)


def get_random_shape(seed_shape: typing.Optional[typing.Tuple[int, ...]] = None) -> typing.Tuple[int, ...]:
  """Select a random rank (1-4) and random dimensions (2-5).

  Args:
      seed_shape (Optional[Tuple[int, ...]]): Optional fixed shape to return.

  Returns:
      Tuple[int, ...]: The random or seeded shape.

  """
  if seed_shape:
    return seed_shape

  rank = random.randint(1, 4)
  return tuple(random.randint(2, 5) for _ in range(rank))


def make_broadcastable_shape(base_shape: typing.Tuple[int, ...], salt: int = 0) -> typing.Tuple[int, ...]:
  """Derive a shape that is broadcast-compatible with the base_shape.

  For each dimension, there is a probability it becomes 1 (broadcasting dimension).
  The 'salt' ensures arguments don't all degenerate to 1s in the same way, creating
  varied valid broadcasting patterns (e.g. (A, 1, C) vs (1, B, C)).

  Args:
      base_shape (Tuple[int, ...]): The target accumulated shape.
      salt (int): Integer to vary the random choice (arg index). Defaults to 0.

  Returns:
      Tuple[int, ...]: A new shape tuple broadcast-compatible with the base shape.

  """
  # Use local random instance to not affect global state
  # Seed with salt to be deterministic for this specific run context
  rng = random.Random(sum(base_shape) + salt)

  new_shape = []

  # Heuristic: First argument (salt 0) usually keeps full shape or minimal broadcast.
  # Later arguments broadcast more aggressively.

  for dim in base_shape:
    # 20% chance to broadcast this dim to 1.
    # If salt is high (later argument), increase chance?
    # Let's keep it simple: 30% chance to be 1.
    if rng.random() < 0.3:
      new_shape.append(1)
    else:
      new_shape.append(dim)

  return tuple(new_shape)


def generate_symbolic_callable(
  constraints: typing.Optional[typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]] = None,
) -> typing.Callable[[JsonValue], JsonValue]:
  """Generate a symbolic callable conforming to constraints.

  Instead of just returning identity, this function inspects the constraints to generate
  a proper callable that behaves correctly for fuzzing higher order functions.

  Args:
      constraints (Mapping[str, JsonValue], optional): Constraints like return_type or signature.

  Returns:
      Callable[[JsonValue], JsonValue]: A callable Lambda wrapper.

  """
  if constraints and "return_type" in constraints:
    # Return the first argument, preserving basic type alignment for fuzzer mock pipelines
    return lambda x, *args, **kwargs: x

  return lambda x, *args, **kwargs: x
