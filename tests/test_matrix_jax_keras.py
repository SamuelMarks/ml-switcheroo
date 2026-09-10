"""Bidirectional JAX and Keras 3 CST Transpilation Tests.

Verifies static source-to-source code generation between JAX/Flax NNX and Keras 3,
including class inheritance, method renaming (__call__ <-> call), and ops.
"""

import ast
from ml_switcheroo import convert


def test_jax_to_keras_module_transpilation() -> None:
  """Tests transpiling a JAX flax.nnx.Module into a Keras Model."""
  jax_code = """import jax.numpy as jnp
from flax import nnx

class KerasTarget(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense = nnx.Linear(32, 10, rngs=rngs)

    def __call__(self, x):
        return self.dense(x)
"""
  converted = convert(jax_code, source="jax", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "keras.Model" in converted or "keras.Layer" in converted
  assert "def call(self, x):" in converted


def test_keras_to_jax_module_transpilation() -> None:
  """Tests transpiling a Keras Model into a JAX flax.nnx.Module."""
  keras_code = """import keras
import keras.layers as layers

class JaxTarget(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(10)

    def call(self, x):
        return self.dense(x)
"""
  converted = convert(keras_code, source="keras", target="jax")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def __call__(self, x):" in converted


def test_jax_to_keras_math_operations() -> None:
  """Tests converting math calls from jax to keras."""
  jax_code = """import jax.numpy as jnp

def ops_fn(x, y):
    return jnp.add(x, y)
"""
  converted = convert(jax_code, source="jax", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def ops_fn(x, y):" in converted
