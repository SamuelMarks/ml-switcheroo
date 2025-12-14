"""
Tests for Class Structure Rewriting: Keras Support.
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

    # Configure Traits for targets AND SOURCES
    self.framework_configs = {
      "tensorflow": {
        "traits": {
          "module_base": "keras.Model",
          "forward_method": "call",
          "requires_super_init": True,
          "strip_magic_args": ["rngs"],
        }
      },
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}},
      "flax_nnx": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
    }

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory():
  """Factory to create rewriters for different source/target pairs."""
  semantics = MockKerasSemantics()

  def create(source, target):
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
  rewriter = rewriter_factory("torch", "tensorflow")
  code = """ 
class MyLayer(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 

    def forward(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(keras.Model):" in result
  assert "def call(self, x):" in result
  assert "forward" not in result


def test_keras_to_torch(rewriter_factory):
  rewriter = rewriter_factory("tensorflow", "torch")
  code = """ 
class MyLayer(keras.Model): 
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
  rewriter = rewriter_factory("tensorflow", "flax_nnx")
  code = """ 
class MyLayer(keras.Model): 
    def call(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(flax.nnx.Module):" in result
  assert "def __call__(self, x):" in result


def test_jax_to_keras(rewriter_factory):
  rewriter = rewriter_factory("flax_nnx", "tensorflow")
  code = """ 
class MyLayer(flax.nnx.Module): 
    def __init__(self, rngs: nnx.Rngs): 
        pass

    def __call__(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyLayer(keras.Model):" in result
  assert "def call(self, x):" in result

  # Verify stripping rngs
  assert "rngs" not in result
  assert "def __init__(self):" in result

  # Verify super injection (Should happen for Keras target)
  assert "super().__init__()" in result


def test_tf_keras_prefix_support(rewriter_factory):
  rewriter = rewriter_factory("tensorflow", "torch")
  code = """ 
class MyModel(tf.keras.Model): 
    def call(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)

  assert "class MyModel(torch.nn.Module):" in result
  assert "def forward(self, x):" in result
