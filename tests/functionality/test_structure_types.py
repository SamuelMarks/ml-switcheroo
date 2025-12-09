"""
Tests for Type Hint Rewriting in StructureRewriter.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockTypeSemantics(SemanticsManager):
  """
  Mock Semantics defining mappings for types.
  """

  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}  # Clean config for default traits behavior

    # Map 'torch.Tensor' -> 'jax.Array'
    self._inject("Tensor", "torch.Tensor", "jax.Array")
    # Map 'torch.float32' -> 'jax.numpy.float32' (Dtype)
    self._inject("float32", "torch.float32", "jax.numpy.float32")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": []}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockTypeSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_param_hint_full_path(rewriter):
  """
  Input: def f(x: torch.Tensor):
  Output: def f(x: jax.Array):
  """
  code = "def f(x: torch.Tensor): pass"
  result = rewrite_code(rewriter, code)
  assert "def f(x: jax.Array):" in result


def test_param_hint_aliased(rewriter):
  """
  Input:
      from torch import Tensor
      def f(x: Tensor):
  Output:
      ...
      def f(x: jax.Array):
  """
  code = """ 
from torch import Tensor
def f(x: Tensor): pass
"""
  result = rewrite_code(rewriter, code)
  assert "def f(x: jax.Array):" in result


def test_return_hint(rewriter):
  code = "def f() -> torch.Tensor: pass"
  result = rewrite_code(rewriter, code)
  assert "-> jax.Array:" in result


def test_nested_hint_list(rewriter):
  code = "def f(x: List[torch.Tensor]): pass"
  result = rewrite_code(rewriter, code)
  assert "List[jax.Array]" in result


def test_ann_assign_rewrite(rewriter):
  code = "x: torch.Tensor = 1"
  result = rewrite_code(rewriter, code)
  assert "x: jax.Array = 1" in result


def test_attribute_hint_rewrite(rewriter):
  code = "def f(d: torch.float32): pass"
  result = rewrite_code(rewriter, code)
  assert "d: jax.numpy.float32" in result


def test_ignore_name_outside_annotation(rewriter):
  code = """ 
from torch import Tensor
def f(): 
    x = Tensor
"""
  result = rewrite_code(rewriter, code)
  assert "x = Tensor" in result
  assert "x = jax.Array" not in result
