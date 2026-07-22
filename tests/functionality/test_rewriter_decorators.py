"""Test suite for the Rewriter Decorators module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType


class MockDecoratorSemantics(SemanticsManager):
  """Mock Decorator Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockDecoratorSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}
    self._inject("jit", "torch.jit.script", "jax.jit")
    self._inject("inference_mode", "torch.inference_mode", None)
    self._inject("compile", "torch.compile", "jax.jit")

  def get_framework_config(self, framework: str):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    """Mock implementation of  inject."""
    variants = {"torch": {"api": s_api}}
    if t_api is None:
      variants["jax"] = None
    else:
      variants["jax"] = {"api": t_api}
    self.data[name] = {"op_type": OpType.DECORATOR, "variants": variants, "std_args": ["fn"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  semantics = MockDecoratorSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter, code):
  """Rewrites ."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewriter failed: {e}")


def test_decorator_renaming(rewriter):
  """Verifies the behavior of decorator renaming."""
  code = "\n@torch.jit.script\ndef func(x):\n    return x\n"
  result = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.jit.script" not in result


def test_decorator_removal(rewriter):
  """Verifies the behavior of decorator removal."""
  code = "\n@torch.inference_mode\ndef func(x):\n    return x\n"
  result = rewrite(rewriter, code)
  assert "@torch.inference_mode" not in result
  assert "def func(x):" in result


def test_call_decorator_renaming(rewriter):
  """Verifies the behavior of call decorator renaming."""
  code = "\n@torch.compile(fullgraph=True)\ndef func(x):\n    pass\n"
  result = rewrite(rewriter, code)
  assert "@jax.jit(fullgraph=True)" in result
  assert "torch.compile" not in result


def test_multiple_decorators_mixed(rewriter):
  """Verifies the behavior of multiple decorators mixed."""
  code = "\n@torch.jit.script\n@torch.inference_mode\ndef f():\n    pass\n"
  result = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.inference_mode" not in result
  assert "def f():" in result
