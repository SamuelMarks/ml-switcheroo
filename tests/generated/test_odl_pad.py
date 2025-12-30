import pytest
import numpy as np
import numpy
import random
import math

# --- flax_nnx Setup ---
try:
  import jax
  import jax.numpy as jnp

  _FLAX_NNX_AVAILABLE = True
except ImportError:
  _FLAX_NNX_AVAILABLE = False

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


def test_gen_Pad():
  # Generated Test for Pad
  # 1. Inputs
  np_input = np.random.randn(2, 2, 2).astype(np.float32)
  np_pad = [1, 2]
  np_mode = "test_val"
  np_value = random.uniform(0.1, 1.0)
  results = {}

  # Framework: flax_nnx
  if _FLAX_NNX_AVAILABLE:
    try:
      # JIT Compilation Check
      fn = lambda a0, a1, a2, a3: jax.numpy.pad(a0, a1, a2, a3)
      jitted_fn = jax.jit(fn, static_argnums=None)
      res = jitted_fn(jnp.array(np_input), np_pad, np_mode, np_value)
      results["flax_nnx"] = np.array(res)
    except Exception as e:
      print(f"Skipping flax_nnx due to error: {e}")

  # Framework: jax
  if JAX_AVAILABLE:
    try:
      # JIT Compilation Check
      fn = lambda a0, a1, a2, a3: jax.numpy.pad(a0, a1, a2, a3)
      jitted_fn = jax.jit(fn, static_argnums=(2,))
      res = jitted_fn(jnp.array(np_input), np_pad, np_mode, np_value)
      results["jax"] = np.array(res)
    except Exception as e:
      print(f"Skipping jax due to error: {e}")

  # Framework: keras
  if _KERAS_AVAILABLE:
    try:
      res = keras.ops.pad(keras.ops.convert_to_tensor(np_input), np_pad, np_mode, np_value)
      results["keras"] = keras.ops.convert_to_numpy(res)
    except Exception as e:
      print(f"Skipping keras due to error: {e}")

  # Framework: mlx
  if _MLX_AVAILABLE:
    try:
      res = mlx.core.pad(mx.array(np_input), np_pad, np_mode, np_value)
      results["mlx"] = np.array(res)
    except Exception as e:
      print(f"Skipping mlx due to error: {e}")

  # Framework: numpy
  if _NUMPY_AVAILABLE:
    try:
      res = numpy.pad(np_input, np_pad, np_mode, np_value)
      results["numpy"] = res
    except Exception as e:
      print(f"Skipping numpy due to error: {e}")

  # Framework: tensorflow
  if TENSORFLOW_AVAILABLE:
    try:
      res = tf.pad(tf.convert_to_tensor(np_input), np_pad, np_mode, np_value)
      results["tensorflow"] = res.numpy()
    except Exception as e:
      print(f"Skipping tensorflow due to error: {e}")

  # Framework: torch
  if TORCH_AVAILABLE:
    try:
      res = torch.nn.functional.pad(torch.tensor(np_input), np_pad, np_mode, np_value)
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
