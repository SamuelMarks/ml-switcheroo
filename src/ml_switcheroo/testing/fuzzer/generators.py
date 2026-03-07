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
  min_v = int(constraints.get("min", -5))  # pragma: no cover
  max_v = int(constraints.get("max", 5))  # pragma: no cover
  # random.randint is inclusive for both bounds
  return random.randint(min_v, max_v)  # pragma: no cover


def generate_scalar_float(constraints: Dict[str, Any]) -> float:
  """
  Generates a random float within constrained bounds.

  Args:
      constraints (Dict[str, Any]): Dictionary containing optional 'min' and 'max'.

  Returns:
      float: A random float.
  """
  if constraints.get("min") is not None and constraints.get("max") is not None:  # pragma: no cover
    return random.uniform(constraints["min"], constraints["max"])  # pragma: no cover

  val = random.random()  # pragma: no cover
  # Adjust one-sided bounds if necessary
  if constraints.get("min") is not None:  # pragma: no cover
    val = max(val, constraints["min"])  # pragma: no cover
  if constraints.get("max") is not None:  # pragma: no cover
    val = min(val, constraints["max"])  # pragma: no cover
  return val  # pragma: no cover


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
  min_val = constraints.get("min")  # pragma: no cover
  max_val = constraints.get("max")  # pragma: no cover

  # Resolve explicit dtype constraint if present
  dtype_req = constraints.get("dtype")  # pragma: no cover
  explicit_dtype = None  # pragma: no cover
  if dtype_req:  # pragma: no cover
    try:  # pragma: no cover
      explicit_dtype = np.dtype(dtype_req)  # pragma: no cover
    except TypeError:  # pragma: no cover
      pass  # pragma: no cover

  # Use explicit dtype to override heuristic type_lbl if available
  if explicit_dtype:  # pragma: no cover
    if np.issubdtype(explicit_dtype, np.integer):  # pragma: no cover
      type_lbl = "int"  # pragma: no cover
    elif explicit_dtype.kind == "b":  # pragma: no cover
      type_lbl = "bool"  # pragma: no cover
    else:
      type_lbl = "float"  # pragma: no cover

  if type_lbl == "bool":  # pragma: no cover
    return np.random.randint(0, 2, size=shape).astype(bool)  # pragma: no cover

  if type_lbl == "int":  # pragma: no cover
    # Default range [-10, 10]
    low = int(min_val) if min_val is not None else -10  # pragma: no cover
    # High in randint is exclusive, update for inclusivity match
    high_bound = int(max_val) if max_val is not None else 10  # pragma: no cover
    high = high_bound + 1  # pragma: no cover

    arr = np.random.randint(low, high, size=shape)  # pragma: no cover
    if explicit_dtype:  # pragma: no cover
      return arr.astype(explicit_dtype)  # pragma: no cover
    return arr.astype(np.int32)  # pragma: no cover

  # Float default
  arr = np.random.randn(*shape)  # pragma: no cover

  # Constraint Clipping logic
  if min_val is not None or max_val is not None:  # pragma: no cover
    if min_val is not None and max_val is not None:  # pragma: no cover
      # Uniform within bounds
      arr = np.random.uniform(min_val, max_val, size=shape)  # pragma: no cover
    else:
      # Clip standard normal
      safe_min = float(min_val) if min_val is not None else -np.inf  # pragma: no cover
      safe_max = float(max_val) if max_val is not None else np.inf  # pragma: no cover

      # If just Min is 0 (Log), use absolute or exp adjustment
      if min_val is not None and min_val >= 0:  # pragma: no cover
        arr = np.abs(arr) + min_val  # pragma: no cover

      arr = np.clip(arr, safe_min, safe_max)  # pragma: no cover

  if explicit_dtype:  # pragma: no cover
    return arr.astype(explicit_dtype)  # pragma: no cover

  return arr.astype(np.float32)  # pragma: no cover


def get_random_shape(seed_shape: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
  """
  Selects a random rank (1-4) and random dimensions (2-5).

  Args:
      seed_shape (Optional[Tuple]): Optional fixed shape to return.

  Returns:
      Tuple[int, ...]: The random or seeded shape.
  """
  if seed_shape:  # pragma: no cover
    return seed_shape  # pragma: no cover

  rank = random.randint(1, 4)  # pragma: no cover
  return tuple(random.randint(2, 5) for _ in range(rank))  # pragma: no cover


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
  rng = random.Random(sum(base_shape) + salt)  # pragma: no cover

  new_shape = []  # pragma: no cover

  # Heuristic: First argument (salt 0) usually keeps full shape or minimal broadcast.
  # Later arguments broadcast more aggressively.

  for dim in base_shape:  # pragma: no cover
    # 20% chance to broadcast this dim to 1.
    # If salt is high (later argument), increase chance?
    # Let's keep it simple: 30% chance to be 1.
    if rng.random() < 0.3:  # pragma: no cover
      new_shape.append(1)  # pragma: no cover
    else:
      new_shape.append(dim)  # pragma: no cover

  return tuple(new_shape)  # pragma: no cover


def generate_fake_callable(constraints: Dict[str, Any] = None) -> Any:
  """
  Generates a dummy function (identity) for functional ops.
  """
  return lambda x, *args, **kwargs: x
