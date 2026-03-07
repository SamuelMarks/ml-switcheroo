"""
String Parsing and Shape Utilities for the Fuzzer.

This module provides helper functions for parsing complex type strings
(e.g., handling nested brackets in generics) and resolving symbolic dimensions
for tensor shapes.
"""

from typing import List, Dict, Tuple
import random


def is_pipe_top_level(text: str) -> bool:
  """
  Checks if a pipe `|` exists outside of brackets `[]`.

  Used to detect Union types (A | B) in Python 3.10+ syntax without
  getting confused by nested types like `List[int | str]`.

  Args:
      text (str): The type hint string to analyze.

  Returns:
      bool: True if a top-level pipe is found.
  """
  depth = 0  # pragma: no cover
  for char in text:  # pragma: no cover
    if char == "[":  # pragma: no cover
      depth += 1  # pragma: no cover
    elif char == "]":  # pragma: no cover
      depth -= 1  # pragma: no cover
    elif char == "|" and depth == 0:  # pragma: no cover
      return True  # pragma: no cover
  return False  # pragma: no cover


def split_outside_brackets(text: str) -> List[str]:
  """
  Splits a string by commas, respecting nested brackets.

  Used to parse generic arguments (e.g., `Tuple[int, List[int]]` -> `['int', 'List[int]']`).

  Args:
      text (str): The content inside brackets.

  Returns:
      List[str]: Split parts.
  """
  parts = []  # pragma: no cover
  current = []  # pragma: no cover
  depth = 0  # pragma: no cover
  for char in text:  # pragma: no cover
    if char == "[":  # pragma: no cover
      depth += 1  # pragma: no cover
      current.append(char)  # pragma: no cover
    elif char == "]":  # pragma: no cover
      depth -= 1  # pragma: no cover
      current.append(char)  # pragma: no cover
    elif char == "," and depth == 0:  # pragma: no cover
      parts.append("".join(current).strip())  # pragma: no cover
      current = []  # pragma: no cover
    else:
      current.append(char)  # pragma: no cover
  if current:  # pragma: no cover
    parts.append("".join(current).strip())  # pragma: no cover
  return parts  # pragma: no cover


def resolve_symbolic_shape(dims_str: str, symbol_map: Dict[str, int]) -> Tuple[int, ...]:
  """
  Parses dimension strings like "'B', 32" or "N, M" into integer tuples.

  Resolves symbolic names using the provided `symbol_map`. If a symbol
  is encountered for the first time, a random size is assigned and stored
  in the map to ensure consistency across arguments.

  Args:
      dims_str (str): The dimensions definition string (e.g. "'N', 32").
      symbol_map (Dict[str, int]): Context for resolving symbols like 'N'.

  Returns:
      Tuple[int, ...]: The concrete shape tuple.
  """
  shape = []  # pragma: no cover
  raw_dims = [d.strip() for d in dims_str.split(",")]  # pragma: no cover

  for dim in raw_dims:  # pragma: no cover
    # Clean possible quotes
    clean = dim.replace("'", "").replace('"', "")  # pragma: no cover
    if not clean:  # pragma: no cover
      continue  # pragma: no cover

    # Check if it's a fixed integer literal (e.g., Array[3, 'C'])
    try:  # pragma: no cover
      val = int(clean)  # pragma: no cover
      shape.append(val)  # pragma: no cover
      continue  # pragma: no cover
    except ValueError:  # pragma: no cover
      pass  # pragma: no cover

    # Validate symbol is a valid identifier or simple char
    if not clean.isidentifier() and not clean.replace("_", "").isalnum():  # pragma: no cover
      # Fallback random for complex expressions we don't parse (e.g. N+1)
      shape.append(random.randint(2, 6))  # pragma: no cover
      continue  # pragma: no cover

    # It's a symbol
    if clean not in symbol_map:  # pragma: no cover
      symbol_map[clean] = random.randint(2, 6)  # pragma: no cover
    shape.append(symbol_map[clean])  # pragma: no cover

  return tuple(shape)  # pragma: no cover


def adjust_shape_rank(shape: Tuple[int, ...], required_rank: int) -> Tuple[int, ...]:
  """
  Adjusts a shape tuple to match a required rank by padding or truncation.

  Args:
      shape (Tuple[int, ...]): The base shape.
      required_rank (int): The target number of dimensions.

  Returns:
      Tuple[int, ...]: The adjusted shape.
  """
  current_rank = len(shape)  # pragma: no cover
  if current_rank == required_rank:  # pragma: no cover
    return shape  # pragma: no cover
  elif current_rank < required_rank:  # pragma: no cover
    # Pad with random dimensions
    extra = tuple(random.randint(2, 5) for _ in range(required_rank - current_rank))  # pragma: no cover
    return shape + extra  # pragma: no cover
  else:
    # Truncate
    return shape[:required_rank]  # pragma: no cover
