"""
Tests for Class Structure Rewriting: Keras Support.

This module verifies the Tri-directional class translation logic including Keras.

Verifies:
    1.  Torch -> Keras: `nn.Module` -> `keras.Layer`, `forward` -> `call`.
    2.  Keras -> Torch: `keras.Layer` -> `nn.Module`, `call` -> `forward`.
    3.  Keras -> JAX: `keras.Layer` -> `flax.nnx.Module`, `call` -> `__call__`.
    4.  JAX -> Keras: `flax.nnx.Module` -> `keras.Layer`, `__call__` -> `call`.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockKerasSemantics(SemanticsManager):
  """
  Mock Manager with mappings for Neural abstractions including Keras.
  """

  def __init__(self) -> None:
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # Define common ops if needed, but StructureMixin mainly relies on
    # hardcoded class patterns and generic method renaming logic.
    pass


@pytest.fixture
def rewriter_factory():
  """Factory to create rewriters for different source/target pairs."""
  semantics = MockKerasSemantics()

  def create(source, target):
    # Warning: target_framework logic inside RuntimeConfig casts to Enum string
    # We use "tensorflow" to represent the Keras target as per SupportedEngine
    config = RuntimeConfig(source_framework=source, target_framework=target, strict_mode=False)
    return PivotRewriter(semantics, config)

  return create


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Helper to parse and rewrite code."""
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewriter crashed: {e}")


def test_torch_to_keras(rewriter_factory):
  """
  Scenario: Convert PyTorch Module to Keras Layer.
  Inheritance: torch.nn.Module -> keras.Layer
  Method: forward -> call
  """
  rewriter = rewriter_factory("torch", "tensorflow")
  code = """ 
class MyLayer(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 

    def forward(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(keras.Layer):" in result
  assert "def call(self, x):" in result
  assert "forward" not in result


def test_keras_to_torch(rewriter_factory):
  """
  Scenario: Convert Keras Layer to PyTorch Module.
  Inheritance: keras.Layer -> torch.nn.Module
  Method: call -> forward
  Init: Ensure super().__init__() is present (Keras may or may not have it)
  """
  rewriter = rewriter_factory("tensorflow", "torch")
  code = """ 
class MyLayer(keras.Layer): 
    def __init__(self): 
        pass

    def call(self, inputs): 
        return inputs
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(torch.nn.Module):" in result
  assert "def forward(self, inputs):" in result
  assert "super().__init__()" in result


def test_keras_to_jax(rewriter_factory):
  """
  Scenario: Convert Keras Layer to Flax NNX Module.
  Inheritance: keras.Layer -> flax.nnx.Module
  Method: call -> __call__
  """
  rewriter = rewriter_factory("tensorflow", "jax")
  code = """ 
class MyLayer(keras.Layer): 
    def call(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(flax.nnx.Module):" in result
  assert "def __call__(self, x):" in result


def test_jax_to_keras(rewriter_factory):
  """
  Scenario: Convert Flax NNX Module to Keras Layer.
  Inheritance: flax.nnx.Module -> keras.Layer
  Method: __call__ -> call
  Init: Strip rngs (if present in input)
  """
  rewriter = rewriter_factory("jax", "tensorflow")
  code = """ 
class MyLayer(flax.nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        pass

    def __call__(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(keras.Layer):" in result
  assert "def call(self, x):" in result

  # Verify stripping rngs
  assert "rngs" not in result
  assert "def __init__(self):" in result

  # Verify super injection (Should happen for Keras target)
  assert "super().__init__()" in result


def test_tf_keras_prefix_support(rewriter_factory):
  """
  Verify `tf.keras.layers.Layer` is detected as a Keras class.
  """
  rewriter = rewriter_factory("tensorflow", "torch")
  code = """ 
class MyModel(tf.keras.Model): 
    def call(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyModel(torch.nn.Module):" in result
  assert "def forward(self, x):" in result
