import pytest
import numpy as np
import numpy
import random
import math

# --- jax Setup ---
try:
  import jax
  import jax.numpy as jnp

  JAX_AVAILABLE = True
except ImportError:
  JAX_AVAILABLE = False

# --- keras Setup ---
try:
  import keras
  from keras import ops

  _KERAS_AVAILABLE = True
except ImportError:
  _KERAS_AVAILABLE = False

# --- mlx Setup ---
try:
  import mlx.core as mx
  import numpy as np

  _MLX_AVAILABLE = True
except ImportError:
  _MLX_AVAILABLE = False

# --- numpy Setup ---
try:
  import numpy as np

  _NUMPY_AVAILABLE = True
except ImportError:
  _NUMPY_AVAILABLE = False

# --- tensorflow Setup ---
try:
  import tensorflow as tf

  TENSORFLOW_AVAILABLE = True
except ImportError:
  TENSORFLOW_AVAILABLE = False

# --- torch Setup ---
try:
  import torch

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


def test_gen_RandInt():
  # Generated Test for RandInt
  # 1. Inputs
  np_low = random.randint(1, 3)
  np_high = random.randint(1, 3)
  np_shape = (1, 2)
  np_dtype = None
  results = {}

  # Framework: jax
  if JAX_AVAILABLE:
    try:
      # JIT Compilation Check
      fn = lambda a0, a1, a2, a3: jax.random.randint(a0, a1, a2, a3)
      jitted_fn = jax.jit(fn, static_argnums=(3,))
      res = jitted_fn(np_low, np_high, np_shape, np_dtype)
      results["jax"] = np.array(res)
    except Exception as e:
      print(f"Skipping jax due to error: {e}")

  # Framework: keras
  if _KERAS_AVAILABLE:
    try:
      res = keras.random.randint(np_low, np_high, np_shape, np_dtype)
      results["keras"] = keras.ops.convert_to_numpy(res)
    except Exception as e:
      print(f"Skipping keras due to error: {e}")

  # Framework: mlx
  if _MLX_AVAILABLE:
    try:
      res = mlx.random.randint(np_low, np_high, np_shape, np_dtype)
      results["mlx"] = np.array(res)
    except Exception as e:
      print(f"Skipping mlx due to error: {e}")

  # Framework: numpy
  if _NUMPY_AVAILABLE:
    try:
      res = numpy.random.randint(np_low, np_high, np_shape, np_dtype)
      results["numpy"] = res
    except Exception as e:
      print(f"Skipping numpy due to error: {e}")

  # Framework: tensorflow
  if TENSORFLOW_AVAILABLE:
    try:
      res = tf.random.uniform(np_low, np_high, np_shape, np_dtype)
      results["tensorflow"] = res.numpy()
    except Exception as e:
      print(f"Skipping tensorflow due to error: {e}")

  # Framework: torch
  if TORCH_AVAILABLE:
    try:
      res = torch.randint(np_low, np_high, np_shape, np_dtype)
      results["torch"] = res.detach().cpu().numpy()
    except Exception as e:
      print(f"Skipping torch due to error: {e}")

  # 5. Comparison
  if len(results) < 2:
    pytest.skip("Not enough successful backends to compare")

  vals = list(results.values())
  ref = vals[0]
  for val in vals[1:]:
    if hasattr(ref, "shape") and hasattr(val, "shape"):
      np.testing.assert_allclose(ref, val, rtol=1e-3, atol=1e-3)
    else:
      assert ref == val
