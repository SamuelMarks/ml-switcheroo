"""Shared runtime flags for generated tests (Auto-Generated)."""
import sys
import pytest
import random
import numpy as np

# --- jax ---
try:
    import jax
    import jax.numpy as jnp
    try:
        import chex
    except ImportError:
        pass
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# --- torch ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Determinism ---
@pytest.fixture(autouse=True)
def ensure_determinism():
  """
  Auto-injects fixed seeds for reproducibility at the start of every test.
  Covers Python random, NumPy, Torch, TensorFlow, and MLX.
  """
  # Core Python & NumPy
  random.seed(42)
  np.random.seed(42)

  # PyTorch
  if "torch" in sys.modules:
    try:
      sys.modules["torch"].manual_seed(42)
      if sys.modules["torch"].cuda.is_available():
        sys.modules["torch"].cuda.manual_seed_all(42)
    except Exception:
      pass

  # TensorFlow
  if "tensorflow" in sys.modules:
    try:
      tf = sys.modules["tensorflow"]
      # TF 2.x
      if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
        tf.random.set_seed(42)
    except Exception:
      pass

  # MLX
  if "mlx.core" in sys.modules:
    try:
      sys.modules["mlx.core"].random.seed(42)
    except Exception:
      pass
  elif "mlx" in sys.modules and hasattr(sys.modules["mlx"], "core"):
    try:
      sys.modules["mlx"].core.random.seed(42)
    except Exception:
      pass

# --- Verification Logic ---
def verify_results(ref, val, rtol=1e-3, atol=1e-3, exact=False):
  """
  Cross-framework comparison helper.

  Recursively compares data structures (Lists, Dicts, Tuples).
  If 'exact' is True, enforces strict equality for all types (np.array_equal).
  If 'exact' is False (default), applies fuzzy matching (np.allclose) for floats/complex.
  """
  # 1. Null/None Check (Exact identity)
  if ref is None or val is None:
    return ref is val

  # 2. Try Chex (Structural comparison for JAX PyTrees)
  if "chex" in globals():
    try:
      chex_mod = globals()["chex"]
      if exact:
        chex_mod.assert_trees_all_close(ref, val, rtol=0, atol=0)
      else:
        chex_mod.assert_trees_all_close(ref, val, rtol=rtol, atol=atol)
      return True
    except (AssertionError, Exception):
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

  # 4. Leaf Node Comparison
  try:
    np_ref = np.asanyarray(ref)
    np_val = np.asanyarray(val)

    if np_ref.shape != np_val.shape:
      # Allow scalar vs 0-d array flexibility
      if not (np_ref.size == 1 and np_val.size == 1):
        return False

    if exact:
      return np.array_equal(np_ref, np_val)

    kind = np_ref.dtype.kind
    if kind in {"f", "c"}:
      # Use equal_nan=True to handle NaNs consistently
      return np.allclose(np_ref, np_val, rtol=rtol, atol=atol, equal_nan=True)

    return np.array_equal(np_ref, np_val)

  except Exception:
    try:
      return ref == val
    except Exception:
      return False
