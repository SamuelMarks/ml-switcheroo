"""
Runtime helpers for generated verification tests.

This module contains the reference implementation of ``verify_results``,
a robust recursive comparison utility for validating equivalence between
arbitrary data structures (Arrays, Lists, Dicts) across different frameworks.

It also defines the ``ensure_determinism`` fixture, which is injected into
generated tests to enforce reproducibility by seeding RNGs for Torch, JAX,
TensorFlow, Numpy, and Python.
"""

from typing import Any

import numpy as np
import random
import sys
import pytest


@pytest.fixture(autouse=True)
def ensure_determinism() -> None:
  """
  Auto-injects fixed seeds for reproducibility at the start of every test.

  Covers:
  - Python `random`
  - NumPy `np.random`
  - PyTorch `torch.manual_seed` (CPU & CUDA)
  - TensorFlow `tf.random.set_seed`
  - MLX `mlx.core.random.seed`
  """
  seed = 42

  # Core Python & NumPy
  random.seed(seed)
  np.random.seed(seed)

  # PyTorch
  if "torch" in sys.modules:
    try:
      sys.modules["torch"].manual_seed(seed)
      if sys.modules["torch"].cuda.is_available():
        sys.modules["torch"].cuda.manual_seed_all(seed)
    except Exception:
      pass

  # TensorFlow
  if "tensorflow" in sys.modules:
    try:
      tf = sys.modules["tensorflow"]
      # TF 2.x
      if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
        tf.random.set_seed(seed)
    except Exception:
      pass

  # MLX
  if "mlx.core" in sys.modules:
    try:
      sys.modules["mlx.core"].random.seed(seed)
    except Exception:
      pass
  elif "mlx" in sys.modules and hasattr(sys.modules["mlx"], "core"):
    try:
      sys.modules["mlx"].core.random.seed(seed)
    except Exception:
      pass


def verify_results(ref: Any, val: Any, rtol: float = 1e-3, atol: float = 1e-4, exact: bool = False) -> bool:
  """
  Cross-framework comparison helper.

  Recursively compares data structures (Lists, Dicts, Tuples, Arrays).

  Modes:
  - **Fuzzy (Default)**: Uses `np.allclose` with tolerances for floats.
  - **Exact**: Enforces strict equality (ids for None, `np.array_equal` for arrays).

  Args:
      ref (Any): The reference value (e.g. from Source Framework).
      val (Any): The candidate value (e.g. from Target Framework).
      rtol (float): Relative tolerance for floating point comparison.
      atol (float): Absolute tolerance for floating point comparison.
      exact (bool): If True, disables fuzzy matching.

  Returns:
      bool: True if values are considered equivalent.
  """
  # 1. Null/None Check (Exact identity)
  if ref is None or val is None:
    return ref is val

  # 2. Try Chex (Structural comparison for JAX PyTrees)
  if "chex" in globals():
    try:
      chex_mod = globals()["chex"]
      # Chex assert functions raise errors on mismatch
      if exact:
        chex_mod.assert_trees_all_close(ref, val, rtol=0, atol=0)
      else:
        chex_mod.assert_trees_all_close(ref, val, rtol=rtol, atol=atol)
      return True
    except (AssertionError, Exception):
      # Fallback to manual recursive comparison if Chex/Tree utils fail
      pass

  # 3. Recursive Container Handling
  if isinstance(ref, dict) and isinstance(val, dict):
    if ref.keys() != val.keys():
      return False
    for k in ref:
      if not verify_results(ref[k], val[k], rtol, atol, exact=exact):
        return False
    return True

  if isinstance(ref, (list, tuple)) and isinstance(val, (list, tuple)):
    if len(ref) != len(val):
      return False
    for r, v in zip(ref, val):
      if not verify_results(r, v, rtol, atol, exact=exact):
        return False
    return True

  # 4. Leaf Node Comparison (Arrays/Scalars)
  try:
    np_ref = np.asanyarray(ref)
    np_val = np.asanyarray(val)

    if np_ref.shape != np_val.shape:
      # Allow scalar vs 0-d array flexibility
      if not (np_ref.size == 1 and np_val.size == 1):
        return False

    if exact:
      return np.array_equal(np_ref, np_val)

    # Determine kind to choose appropriate comparison
    kind = np_ref.dtype.kind

    # Float (f) or Complex (c) -> Fuzzy Match
    if kind in {"f", "c"}:
      return np.allclose(np_ref, np_val, rtol=rtol, atol=atol, equal_nan=True)

    # Integer/Bool/String -> Exact Match
    return np.array_equal(np_ref, np_val)

  except Exception:
    # Fallback for types that fail numpy conversion (e.g. custom objects)
    try:
      return ref == val
    except Exception:
      return False
