"""
Tests for Trait-Based Structural Rewriting using TestRewriter.
"""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks import register_framework


class MockTraitSemantics(SemanticsManager):
  """
  Mock Manager that returns explicit traits.
  """

  def __init__(self):
    # Bypass super().__init__ which loads files
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}
    self.test_templates = {}

    self.framework_configs = {
      "custom_nn": {
        "traits": {
          "module_base": "custom.Layer",
          "forward_method": "predict",
          "requires_super_init": True,
          "init_method_name": "__init__",
          # Note: Both are defined, so result will have ctx AND stripped rngs
          "inject_magic_args": [("ctx", "custom.Context")],
          "strip_magic_args": ["rngs"],
        }
      },
      "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "requires_super_init": True,
        }
      },
      "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
    }

  def get_framework_config(self, framework: str) -> dict:
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory():
  # Register dummy adapter for 'custom_nn' so RuntimeConfig validation passes during the test
  class CustomNNAdapter:
    def convert(self, x):
      return x

  register_framework("custom_nn")(CustomNNAdapter)
  register_framework("vanilla")(CustomNNAdapter)
  register_framework("ghost_fw")(CustomNNAdapter)

  semantics = MockTraitSemantics()

  def create(target_fw):
    config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=False)
    return PivotRewriter(semantics, config)

  return create


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_trait_module_inheritance_rewrite(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = "class Model(torch.nn.Module): pass"
  result = rewrite_code(rewriter, code)
  assert "class Model(custom.Layer):" in result


def test_dynamic_base_discovery(rewriter_factory):
  """
  Verifies that a completely unknown framework base ('ghost.Network')
  is detected as a Module purely because it exists in the SemanticsManager config.
  """
  semantics = MockTraitSemantics()
  config = RuntimeConfig(source_framework="ghost_fw", target_framework="custom_nn", strict_mode=False)
  rewriter = PivotRewriter(semantics, config)

  code = """
class MyGhost(ghost.Network):
    def forward(self, x):
        pass
"""
  result = rewrite_code(rewriter, code)

  assert "class MyGhost(custom.Layer):" in result
  assert "def predict(self, x):" in result


def test_trait_method_renaming(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = """
class Model(torch.nn.Module):
    def forward(self, x):
        pass
"""
  result = rewrite_code(rewriter, code)
  assert "def predict(self, x):" in result
  assert "def forward" not in result


def test_trait_argument_injection(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = "class Model(torch.nn.Module): \n    def __init__(self): pass"
  result = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context):" in result


def test_trait_super_init_requirement(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = """
class Model(torch.nn.Module):
    def __init__(self):
        self.x = 1
"""
  result = rewrite_code(rewriter, code)
  assert "super().__init__()" in result


def test_trait_arg_stripping(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = """
class Model(torch.nn.Module):
    def __init__(self, rngs, x):
        pass
"""
  result = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context, x):" in result
  assert "rngs" not in result
