"""Test suite for the Structure module."""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.framework_configs = {
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
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
    self.data["Tensor"] = {"variants": {"jax": {"api": "jax.Array"}}}
    self._reverse_index = {"torch.Tensor": ("Tensor", self.data["Tensor"])}

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(fw, {})

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, aid, fw):
    """Mock implementation of resolve variant."""
    if aid in self.data and fw in self.data[aid].get("variants", {}):
      return self.data[aid]["variants"][fw]
    return None


@pytest.fixture
def run_pass():
  """Provides a mock run pass for testing."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  context = RewriterContext(semantics, config)
  context.alias_map["torch"] = "torch"
  context.alias_map["torch.nn"] = "torch.nn"

  def _transform(code):
    """Helper to  transform."""
    module = cst.parse_module(code)
    struct_pass = StructuralPass()
    return struct_pass.transform(module, context).code

  return _transform


def test_class_base_rewrite(run_pass):
  """Verifies the behavior of class base rewrite."""
  code = "class Net(torch.nn.Module): pass"
  res = run_pass(code)
  assert "class Net(flax.nnx.Module):" in res


def test_class_base_rewrite_aliased(run_pass):
  """Verifies the behavior of class base rewrite aliased."""
  code = "class Net(torch.nn.Module): pass"
  res = run_pass(code)
  assert "flax.nnx.Module" in res


def test_method_renaming(run_pass):
  """Verifies the behavior of method renaming."""
  code = "\nclass Net(torch.nn.Module):\n    def forward(self, x): pass\n"
  res = run_pass(code)
  assert "def __call__(self, x):" in res


def test_magic_arg_injection(run_pass):
  """Verifies the behavior of magic argument injection."""
  code = "\nclass Net(torch.nn.Module):\n    def __init__(self, dim): pass\n"
  res = run_pass(code)
  assert "def __init__(self, rngs: nnx.Rngs, dim):" in res


def test_super_init_stripping(run_pass):
  """Verifies the behavior of super initialization stripping."""
  code = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.x = 1\n"
  res = run_pass(code)
  assert "super().__init__()" not in res
  assert "self.x = 1" in res


def test_type_hint_rewrite(run_pass):
  """Verifies the behavior of type hint rewrite."""
  code = "def f(x: torch.Tensor): pass"
  res = run_pass(code)
  assert "x: jax.Array" in res


def test_ignore_non_module_classes(run_pass):
  """Verifies the behavior of ignore non module classes."""
  code = "\nclass Data:\n    def forward(self): pass\n"
  res = run_pass(code)
  assert "class Data:" in res
  assert "def forward(self):" in res
