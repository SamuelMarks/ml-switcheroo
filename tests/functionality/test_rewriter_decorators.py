"""
Tests for Decorator Rewriting Logic.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockDecoratorSemantics(SemanticsManager):
  """
  Mock Manager for decorator scenarios.
  """

  def __init__(self):
    # Skip init to avoid file load
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}  # Clean config

    # 1. Rename: torch.jit.script -> jax.jit
    self._inject("jit", "torch.jit.script", "jax.jit")

    # 2. Remove: torch.inference_mode -> None (for JAX)
    self._inject("inference_mode", "torch.inference_mode", None)

    # 3. Call-style: torch.compile -> jax.jit (with args preserved implicitly)
    self._inject("compile", "torch.compile", "jax.jit")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    variants = {"torch": {"api": s_api}}
    if t_api is None:
      variants["jax"] = None  # Explicit removal
    else:
      variants["jax"] = {"api": t_api}

    self.data[name] = {"variants": variants, "std_args": ["fn"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockDecoratorSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite(rewriter, code):
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_decorator_renaming(rewriter):
  code = """ 
@torch.jit.script
def func(x): 
    return x
"""
  result = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.jit.script" not in result


def test_decorator_removal(rewriter):
  code = """ 
@torch.inference_mode
def func(x): 
    return x
"""
  result = rewrite(rewriter, code)
  assert "@torch.inference_mode" not in result
  assert "def func(x):" in result


def test_call_decorator_renaming(rewriter):
  code = """ 
@torch.compile(fullgraph=True) 
def func(x): 
    pass
"""
  result = rewrite(rewriter, code)
  assert "@jax.jit(fullgraph=True)" in result
  assert "torch.compile" not in result


def test_multiple_decorators_mixed(rewriter):
  code = """ 
@torch.jit.script
@torch.inference_mode
def f(): 
    pass
"""
  result = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.inference_mode" not in result
  assert "def f():" in result


def test_unmapped_decorator_preserved(rewriter):
  code = """ 
@unknown.decorator
def f(): pass
"""
  result = rewrite(rewriter, code)
  assert "@unknown.decorator" in result
