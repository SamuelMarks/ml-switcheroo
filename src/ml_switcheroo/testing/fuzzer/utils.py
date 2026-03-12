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
  depth = 0
  for char in text:
    if char == "[":
      depth += 1
    elif char == "]":
      depth -= 1
    elif char == "|" and depth == 0:
      return True
  return False


def split_outside_brackets(text: str) -> List[str]:
  """
  Splits a string by commas, respecting nested brackets.

  Used to parse generic arguments (e.g., `Tuple[int, List[int]]` -> `['int', 'List[int]']`).

  Args:
      text (str): The content inside brackets.

  Returns:
      List[str]: Split parts.
  """
  parts = []
  current = []
  depth = 0
  for char in text:
    if char == "[":
      depth += 1
      current.append(char)
    elif char == "]":
      depth -= 1
      current.append(char)
    elif char == "," and depth == 0:
      parts.append("".join(current).strip())
      current = []
    else:
      current.append(char)
  if current:
    parts.append("".join(current).strip())
  return parts


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
  shape = []
  raw_dims = [d.strip() for d in dims_str.split(",")]

  for dim in raw_dims:
    # Clean possible quotes
    clean = dim.replace("'", "").replace('"', "")
    if not clean:
      continue

    # Check if it's a fixed integer literal (e.g., Array[3, 'C'])
    try:
      val = int(clean)
      shape.append(val)
      continue
    except ValueError:
      pass

    # Validate symbol is a valid identifier or simple char
    if not clean.isidentifier() and not clean.replace("_", "").isalnum():
      # Fallback random for complex expressions we don't parse (e.g. N+1)
      shape.append(random.randint(2, 6))
      continue

    # It's a symbol
    if clean not in symbol_map:
      symbol_map[clean] = random.randint(2, 6)
    shape.append(symbol_map[clean])

  return tuple(shape)


def adjust_shape_rank(shape: Tuple[int, ...], required_rank: int) -> Tuple[int, ...]:
  """
  Adjusts a shape tuple to match a required rank by padding or truncation.

  Args:
      shape (Tuple[int, ...]): The base shape.
      required_rank (int): The target number of dimensions.

  Returns:
      Tuple[int, ...]: The adjusted shape.
  """
  current_rank = len(shape)
  if current_rank == required_rank:
    return shape
  elif current_rank < required_rank:
    # Pad with random dimensions
    extra = tuple(random.randint(2, 5) for _ in range(required_rank - current_rank))
    return shape + extra
  else:
    # Truncate
    return shape[:required_rank]
