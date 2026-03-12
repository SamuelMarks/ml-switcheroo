"""
Primitive Generation Logic for the Fuzzer.

This module contains functions to generate random scalars, NumPy arrays,
and dummy callables respecting specified constraints (min, max, dtype).

Feature Update: Added broadcasting logic generator.
"""

import random
from typing import Any, Dict, Tuple, Optional
import numpy as np


def generate_scalar_int(constraints: Dict[str, Any]) -> int:
  """
  Generates a random integer within constrained bounds.

  Args:
      constraints (Dict[str, Any]): Dictionary containing optional 'min' and 'max'.

  Returns:
      int: A random integer.
  """
  min_v = int(constraints.get("min", -5))
  max_v = int(constraints.get("max", 5))
  # random.randint is inclusive for both bounds
  return random.randint(min_v, max_v)


def generate_scalar_float(constraints: Dict[str, Any]) -> float:
  """
  Generates a random float within constrained bounds.

  Args:
      constraints (Dict[str, Any]): Dictionary containing optional 'min' and 'max'.

  Returns:
      float: A random float.
  """
  if constraints.get("min") is not None and constraints.get("max") is not None:
    return random.uniform(constraints["min"], constraints["max"])

  val = random.random()
  # Adjust one-sided bounds if necessary
  if constraints.get("min") is not None:
    val = max(val, constraints["min"])
  if constraints.get("max") is not None:
    val = min(val, constraints["max"])
  return val


def generate_array(type_lbl: str, shape: Tuple[int, ...], constraints: Dict[str, Any]) -> np.ndarray:
  """
  Generates a random NumPy array bounded by constraints.

  Args:
      type_lbl (str): General type category ('float', 'int', 'bool').
      shape (Tuple[int, ...]): The shape of the array.
      constraints (Dict[str, Any]): Constraints like 'min', 'max', 'dtype'.

  Returns:
      np.ndarray: The generated array.
  """
  # Resolve bounds
  min_val = constraints.get("min")
  max_val = constraints.get("max")

  # Resolve explicit dtype constraint if present
  dtype_req = constraints.get("dtype")
  explicit_dtype = None
  if dtype_req:
    try:
      explicit_dtype = np.dtype(dtype_req)
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

    arr = np.random.randint(low, high, size=shape)
    if explicit_dtype:
      return arr.astype(explicit_dtype)
    return arr.astype(np.int32)

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


def get_random_shape(seed_shape: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
  """
  Selects a random rank (1-4) and random dimensions (2-5).

  Args:
      seed_shape (Optional[Tuple]): Optional fixed shape to return.

  Returns:
      Tuple[int, ...]: The random or seeded shape.
  """
  if seed_shape:
    return seed_shape

  rank = random.randint(1, 4)
  return tuple(random.randint(2, 5) for _ in range(rank))


def make_broadcastable_shape(base_shape: Tuple[int, ...], salt: int = 0) -> Tuple[int, ...]:
  """
  Derives a shape that is broadcast-compatible with the base_shape.

  For each dimension, there is a probability it becomes 1 (broadcasting dimension).
  The 'salt' ensures arguments don't all degenerate to 1s in the same way, creating
  varied valid broadcasting patterns (e.g. (A, 1, C) vs (1, B, C)).

  Args:
      base_shape: The target accumulated shape.
      salt: Integer to vary the random choice (arg index).

  Returns:
      A new shape tuple.
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


def generate_fake_callable(constraints: Dict[str, Any] = None) -> Any:
  """
  Generates a dummy function (identity) for functional ops.
  """
  return lambda x, *args, **kwargs: x
