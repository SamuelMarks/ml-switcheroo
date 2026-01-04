"""
Hypothesis Strategies for ODL Types.

This module maps Operation Definition Language (ODL) type strings (e.g., ``Array['N']``,
``List[int]``) into executable Hypothesis search strategies. It handles:

1.  **Primitives**: Constraints-aware generation for ints, floats, bools.
2.  **Tensors**: Numpy array generation with specific dtypes, ranks, and symbolic shapes.
3.  **Containers**: Recursive generation of Lists, Tuples, and Dictionaries.
4.  **Symbolic Consistency**: Ensuring named dimensions (e.g., 'N') resolve consistently
    across different arguments using a shared context.
"""

import re
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
from typing import Any, Dict, Optional


def _get_dtype_strategy(dtype_str: Optional[str]) -> Any:
  """
  Resolves a string dtype representation to a Numpy dtype or type class.

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
  type_str: str,
  constraints: Dict[str, Any],
  shared_dims: Optional[Dict[str, Any]] = None,
) -> st.SearchStrategy:
  """
  Constructs a Hypothesis strategy from a type string and constraints.

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

  t_clean = str(type_str).strip()

  # 0. Union Types (A | B)
  # Check for top-level pipes.
  if "|" in t_clean:
    # Simple parser for top-level split respecting brackets
    parts = []
    current = []
    depth = 0
    is_union = False
    for char in t_clean:
      if char == "[":
        depth += 1
        current.append(char)
      elif char == "]":
        depth -= 1
        current.append(char)
      elif char == "|" and depth == 0:
        is_union = True
        parts.append("".join(current).strip())
        current = []
      else:
        current.append(char)
    if current:
      parts.append("".join(current).strip())

    if is_union:
      return st.one_of(*[strategies_from_spec(p, constraints, shared_dims) for p in parts])

  # 1. Arrays / Tensors
  if t_clean.startswith(("Array", "Tensor", "np.ndarray")):
    return _array_strategy(t_clean, constraints, shared_dims)

  # 2. Primitives
  if t_clean in ("int", "integer"):
    mn = constraints.get("min")
    mx = constraints.get("max")
    return st.integers(min_value=mn, max_value=mx)

  if t_clean in ("float", "double", "number"):
    mn = float(constraints.get("min", -1e3))
    mx = float(constraints.get("max", 1e3))
    return st.floats(min_value=mn, max_value=mx, allow_nan=False, allow_infinity=False)

  if t_clean in ("bool", "boolean"):
    return st.booleans()

  if t_clean in ("str", "string"):
    # Exclude surrogate characters for safety
    return st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=10)

  if t_clean in ("Callable", "func", "function") or t_clean.startswith("Callable"):
    # Return identity function for functional placeholders
    return st.just(lambda x, *args, **kwargs: x)

  # 3. Containers (Recursive)
  match_opt = re.match(r"^Optional\[(.*)\]$", t_clean)
  if match_opt:
    inner = match_opt.group(1)
    return st.one_of(st.none(), strategies_from_spec(inner, constraints, shared_dims))

  match_list = re.match(r"^List\[(.*)\]$", t_clean)
  if match_list:
    inner = match_list.group(1)
    return st.lists(strategies_from_spec(inner, constraints, shared_dims), min_size=1, max_size=4)

  match_tup_var = re.match(r"^Tuple\[(.*),\s*\.\.\.\]$", t_clean)
  if match_tup_var:
    inner = match_tup_var.group(1)
    return st.lists(strategies_from_spec(inner, constraints, shared_dims), min_size=1, max_size=4).map(tuple)

  match_tup_fixed = re.match(r"^Tuple\[(.*)\]$", t_clean)
  if match_tup_fixed:
    subs = [s.strip() for s in match_tup_fixed.group(1).split(",")]
    sub_strats = [strategies_from_spec(s, constraints, shared_dims) for s in subs]
    return st.tuples(*sub_strats)

  match_dict = re.match(r"^(Dict|Mapping)\[(.*)\]$", t_clean)
  if match_dict:
    inner = match_dict.group(2)
    parts = inner.split(",")
    if len(parts) >= 2:
      k_ref = parts[0].strip()
      v_ref = ",".join(parts[1:]).strip()

      key_strat = strategies_from_spec(k_ref, constraints, shared_dims)
      # FORCE unhashable types (Arrays, Dicts, Lists) to strings to be valid dictionary keys
      if k_ref.startswith(("Array", "Tensor", "np.ndarray")) or "List" in k_ref or "Dict" in k_ref or "Mapping" in k_ref:
        key_strat = key_strat.map(str)

      val_strat = strategies_from_spec(v_ref, constraints, shared_dims)

      return st.dictionaries(
        keys=key_strat,
        values=val_strat,
        min_size=1,
        max_size=3,
      )

  # Dtype Objects
  if "dtype" in t_clean.lower():
    return st.sampled_from([np.float32, np.int32, np.float64, np.bool_])

  # Inference fallback
  if "default" in constraints:
    return st.just(constraints["default"])

  # Fallback default
  return _array_strategy("Array", constraints, shared_dims)


def _array_strategy(type_str: str, constraints: Dict, shared_dims: Optional[Dict]) -> st.SearchStrategy:
  """
  Constructs a numpy array strategy based on rank, symbolic shape, and element constraints.

  Args:
      type_str: Type string (e.g. "Array['N']").
      constraints: User defined constraints (dtype, min, max).
      shared_dims: Dictionary for resolving shared symbolic dimensions.

  Returns:
      A strategy generating np.ndarray.
  """
  dtype = _get_dtype_strategy(constraints.get("dtype"))

  dims = None
  match_sym = re.match(r"^(Array|Tensor)(?:\[(.*)\])?", type_str)

  if match_sym and match_sym.group(2):
    dims_str = match_sym.group(2)
    dims = []
    for d in dims_str.split(","):
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
    min_v = float(mn) if mn is not None else -10.0
    max_v = float(mx) if mx is not None else 10.0
    elements = st.floats(
      min_value=min_v,
      max_value=max_v,
      allow_nan=False,
      allow_infinity=False,
    )

  return npst.arrays(dtype, shape=shape_strat, elements=elements)
