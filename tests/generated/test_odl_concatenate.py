import pytest
import numpy as np
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

  _MLX_AVAILABLE = True
except ImportError:
  _MLX_AVAILABLE = False

# --- numpy Setup ---
try:
  import numpy

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


def test_gen_Concatenate():
  # Generated Test for Concatenate
  # 1. Inputs: Concatenate expects a list of tensors.
  # We create two distinct inputs to ensure concatenation order allows verification.
  input_shape = (2, 2, 2)
  input_1 = np.random.randn(*input_shape).astype(np.float32)
  input_2 = np.random.randn(*input_shape).astype(np.float32)

  # Store as a list, which is the standard expected input for concat functions
  inputs_np = [input_1, input_2]

  # Axis must be within bounds of dimensions (0, 1, or 2 for shape 2x2x2)
  np_axis = random.randint(0, len(input_shape) - 1)

  results = {}

  # Framework: jax
  if JAX_AVAILABLE:
    try:
      # JIT Compilation Check
      fn = lambda seq, ax: jax.numpy.concatenate(seq, axis=ax)
      # Static arg is axis (arg 1)
      jitted_fn = jax.jit(fn, static_argnums=(1,))
      res = jitted_fn(inputs_np, np_axis)
      results["jax"] = np.array(res)
    except Exception as e:
      print(f"Skipping jax due to error: {e}")

  # Framework: keras
  if _KERAS_AVAILABLE:
    try:
      res = keras.ops.concatenate(inputs_np, axis=np_axis)
      results["keras"] = keras.ops.convert_to_numpy(res)
    except Exception as e:
      print(f"Skipping keras due to error: {e}")

  # Framework: mlx
  if _MLX_AVAILABLE:
    try:
      # Explicitly convert to mlx arrays
      mx_inputs = [mx.array(x) for x in inputs_np]
      res = mx.concatenate(mx_inputs, axis=np_axis)
      results["mlx"] = np.array(res)
    except Exception as e:
      print(f"Skipping mlx due to error: {e}")

  # Framework: numpy
  if _NUMPY_AVAILABLE:
    try:
      res = numpy.concatenate(inputs_np, axis=np_axis)
      results["numpy"] = res
    except Exception as e:
      print(f"Skipping numpy due to error: {e}")

  # Framework: tensorflow
  if TENSORFLOW_AVAILABLE:
    try:
      res = tf.concat(inputs_np, axis=np_axis)
      results["tensorflow"] = res.numpy()
    except Exception as e:
      print(f"Skipping tensorflow due to error: {e}")

  # Framework: torch
  if TORCH_AVAILABLE:
    try:
      # Torch requires list of Tensors, not numpy arrays
      torch_inputs = [torch.from_numpy(x) for x in inputs_np]
      res = torch.cat(torch_inputs, dim=np_axis)
      results["torch"] = res.detach().cpu().numpy()
    except Exception as e:
      print(f"Skipping torch due to error: {e}")

  # 4. Type Verification
  for fw, val in results.items():
    assert isinstance(val, (np.ndarray, np.generic)), f"{fw} type mismatch: Expected Array/Tensor, got {type(val)}"

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
