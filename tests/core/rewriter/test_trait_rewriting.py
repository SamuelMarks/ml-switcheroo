"""
Tests for Trait-Based Structural Rewriting.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.testing.adapters import register_adapter


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
      # Define minimal traits for torch to prove injection works when asked
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "requires_super_init": True,
        }
      },
      # A framework unknown to the literal code, added purely via config
      "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
    }

  def get_framework_config(self, framework: str) -> dict:
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory():
  # Register dummy adapter for 'custom_nn' so RuntimeConfig validation passes during the test
  # We do this INSIDE the fixture so it happens per-test, and is cleaned up by the
  # autouse isolation fixture in conftest.py
  class CustomNNAdapter:
    def convert(self, x):
      return x

  register_adapter("custom_nn", CustomNNAdapter)
  register_adapter("vanilla", CustomNNAdapter)  # For fallback test
  register_adapter("ghost_fw", CustomNNAdapter)  # For dynamic detection test

  # sematics setup
  semantics = MockTraitSemantics()

  def create(target_fw):
    config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=False)
    return PivotRewriter(semantics, config)

  return create


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_trait_module_inheritance_rewrite(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = "class Model(torch.nn.Module): pass"
  result = rewrite_code(rewriter, code)
  assert "class Model(custom.Layer):" in result


def test_dynamic_base_discovery(rewriter_factory):
  """
  Verifies that a completely unknown framework base ('ghost.Network')
  is detected as a Module purely because it exists in the SemanticsManager config,
  without hardcoding.
  """
  # We want to convert FROM Ghost FW to Custom NN
  semantics = MockTraitSemantics()
  config = RuntimeConfig(source_framework="ghost_fw", target_framework="custom_nn", strict_mode=False)
  rewriter = PivotRewriter(semantics, config)

  # 2. Input code uses the Ghost Framework base class
  # We use 'forward' as method name to trigger renaming logic, proving Class detection
  code = """
class MyGhost(ghost.Network):
    def forward(self, x):
        pass
"""
  result = rewrite_code(rewriter, code)

  # 3. Assert it was detected as a module and rewritten
  # Base should swap to custom.Layer (from custom_nn traits)
  assert "class MyGhost(custom.Layer):" in result

  # Method should rename to 'predict' (from custom_nn traits)
  # This proves _is_framework_base returned True for "ghost.Network"
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
  # Must match formatting produced by structure_func.py (comma handling)
  # With recent "clean last comma" fix, it should be clean.
  assert "def __init__(self, ctx: custom.Context):" in result


def test_trait_super_init_requirement(rewriter_factory):
  rewriter = rewriter_factory("custom_nn")
  code = """ 
class Model(torch.nn.Module): 
    def __init__(self): 
        self.x = 1 
"""
  result = rewrite_code(rewriter, code)
  # 'requires_super_init' is True for custom_nn
  assert "super().__init__()" in result


def test_trait_arg_stripping(rewriter_factory):
  """
  Verify behavior when stripping 'rngs' and injecting 'ctx'.
  """
  rewriter = rewriter_factory("custom_nn")
  code = """ 
class Model(torch.nn.Module): 
    def __init__(self, rngs, x): 
        pass 
"""
  result = rewrite_code(rewriter, code)

  # 'rngs' should be gone. 'ctx' should be added.
  # Expected: def __init__(self, ctx: custom.Context, x):
  assert "def __init__(self, ctx: custom.Context, x):" in result
  assert "rngs" not in result


def test_no_legacy_defaults_if_missing(rewriter_factory):
  """
  Verify that if a framework has NO traits defined, the rewriter behaves as identity.
  This confirms _LEGACY_DEFAULTS are gone from source.
  """
  # 'vanilla' is registered but has no entry in MockTraitSemantics
  rewriter = rewriter_factory("vanilla")

  code = """ 
class Model(torch.nn.Module): 
    def __init__(self): 
        self.x = 1 
    def forward(self): 
        pass 
"""
  result = rewrite_code(rewriter, code)

  # Assert code did NOT change
  assert "super().__init__()" not in result  # No injection
  assert "def forward" in result  # No rename
