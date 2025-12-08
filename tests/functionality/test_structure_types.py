"""
Tests for Type Hint Rewriting in StructureRewriter.

Verifies that:
1. Function parameter type hints are rewritten (x: torch.Tensor -> x: jax.Array).
2. Return type hints are rewritten (-> torch.Tensor -> -> jax.Array).
3. Variable annotations (assignments) are rewritten (x: torch.Tensor = ...).
4. Imported aliases used as types are resolved (x: Tensor -> x: jax.Array).
5. Generic nested types are handled (List[Tensor] -> List[jax.Array]).
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

    # Map 'torch.Tensor' -> 'jax.Array'
    # We treat types just like API attributes in the semantics structure
    self._inject("Tensor", "torch.Tensor", "jax.Array")

    # Map 'torch.float32' -> 'jax.numpy.float32' (Dtype)
    self._inject("float32", "torch.float32", "jax.numpy.float32")

  def _inject(self, name, s_api, t_api):
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": []}  # Types have no args
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
  """
  Input: def f() -> torch.Tensor:
  Output: def f() -> jax.Array:
  """
  code = "def f() -> torch.Tensor: pass"
  result = rewrite_code(rewriter, code)
  assert "-> jax.Array:" in result


def test_nested_hint_list(rewriter):
  """
  Input: def f(x: List[torch.Tensor]):
  Output: def f(x: List[jax.Array]):
  """
  code = "def f(x: List[torch.Tensor]): pass"
  result = rewrite_code(rewriter, code)
  assert "List[jax.Array]" in result


def test_ann_assign_rewrite(rewriter):
  """
  Input: x: torch.Tensor = torch.randn(1)
  Output: x: jax.Array = ...
  """
  # Note: The RHS 'torch.randn' is not mapped in MockTypeSemantics, so it won't change
  # unless we stick to strict mode, but strict is off by default in test config.
  code = "x: torch.Tensor = 1"
  result = rewrite_code(rewriter, code)
  assert "x: jax.Array = 1" in result


def test_attribute_hint_rewrite(rewriter):
  """
  Input: def f(dtype: torch.float32):
  Output: def f(dtype: jax.numpy.float32):
  """
  code = "def f(d: torch.float32): pass"
  result = rewrite_code(rewriter, code)
  assert "d: jax.numpy.float32" in result


def test_ignore_name_outside_annotation(rewriter):
  """
  Input:
      from torch import Tensor
      x = Tensor()
  Output:
      ...
      x = Tensor()

  (Without Call map for Tensor, 'Tensor' variable name/call should NOT be rewritten
   by StructureMixin logic, only by CallMixin if mapped. Since mock only maps it as type
   and StructureMixin only touches annotations, the body should stay 'Tensor()' if
   CallMixin doesn't recognize it or if we specifically test StructureMixin isolation).
  """
  # We test that leave_Name in StructureMixin guards against non-annotation usage.
  code = """ 
from torch import Tensor
def f(): 
    x = Tensor
"""
  result = rewrite_code(rewriter, code)
  # The 'Tensor' in assignment Rhs is a Name node.
  # Since it's not in an annotation, StructureMixin.leave_Name should skip it.
  # CallMixin/AttrMixin might handle it if programmed, but here we check it's not 'jax.Array'
  # purely via StructureMixin.
  assert "x = Tensor" in result
  assert "x = jax.Array" not in result
