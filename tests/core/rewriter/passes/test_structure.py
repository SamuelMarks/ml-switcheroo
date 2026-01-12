"""
Tests for the Structural Transformation Pass.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import StructuralTraits


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.framework_configs = {
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "forward_method": "forward",
        }
      },
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "init_method_name": "__init__",
          "inject_magic_args": [("rngs", "nnx.Rngs")],
          "requires_super_init": False,
        },
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
    }

    # Mock Type Definition
    self.data["Tensor"] = {"variants": {"jax": {"api": "jax.Array"}}}
    self._reverse_index = {"torch.Tensor": ("Tensor", self.data["Tensor"])}

  def get_framework_config(self, fw):
    return self.framework_configs.get(fw, {})

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def resolve_variant(self, aid, fw):
    if aid in self.data and fw in self.data[aid].get("variants", {}):
      return self.data[aid]["variants"][fw]
    return None


@pytest.fixture
def run_pass():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  # Initialize context directly
  context = RewriterContext(semantics, config)

  # Pre-hydrate aliases for the test context since file loader isn't running
  # This allows _get_qualified_name to find 'torch.Tensor'
  context.alias_map["torch"] = "torch"
  context.alias_map["torch.nn"] = "torch.nn"

  def _transform(code):
    module = cst.parse_module(code)
    struct_pass = StructuralPass()
    return struct_pass.transform(module, context).code

  return _transform


def test_class_base_rewrite(run_pass):
  """Verify class inheritance swap."""
  code = "class Net(torch.nn.Module): pass"
  res = run_pass(code)
  assert "class Net(flax.nnx.Module):" in res


def test_class_base_rewrite_aliased(run_pass):
  """Verify class inheritance swap with alias."""
  # Context alias map must be populated for this to work in real engine.
  # Here we rely on explicit full name or mock behavior.
  code = "class Net(torch.nn.Module): pass"
  res = run_pass(code)
  assert "flax.nnx.Module" in res


def test_method_renaming(run_pass):
  """Verify forward -> __call__."""
  code = """ 
class Net(torch.nn.Module): 
    def forward(self, x): pass
"""
  res = run_pass(code)
  assert "def __call__(self, x):" in res


def test_magic_arg_injection(run_pass):
  """Verify rngs injection."""
  code = """ 
class Net(torch.nn.Module): 
    def __init__(self, dim): pass
"""
  res = run_pass(code)
  assert "def __init__(self, rngs: nnx.Rngs, dim):" in res


def test_super_init_stripping(run_pass):
  """Verify super().__init__() removal for Flax."""
  code = """ 
class Net(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.x = 1
"""
  res = run_pass(code)
  assert "super().__init__()" not in res
  assert "self.x = 1" in res


def test_type_hint_rewrite(run_pass):
  """Verify torch.Tensor -> jax.Array."""
  code = "def f(x: torch.Tensor): pass"
  res = run_pass(code)
  assert "x: jax.Array" in res


def test_ignore_non_module_classes(run_pass):
  """Verify standard classes are untouched."""
  code = """ 
class Data: 
    def forward(self): pass
"""
  res = run_pass(code)
  # Should not be rewritten
  assert "class Data:" in res
  assert "def forward(self):" in res
