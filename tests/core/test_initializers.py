"""
Tests for Initializer Rewriter Mixin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter

# Import the Mixin to ensure it's loaded (PivotRewriter dynamically inherits or we patch)
from ml_switcheroo.core.rewriter.initializers import InitializerMixin
from ml_switcheroo.config import RuntimeConfig


# Create a test class mixing in the Initializer logic
class TestRewriter(InitializerMixin, cst.CSTTransformer):
  def __init__(self):
    self.ctx = MagicMock()
    # Mock expected context props if referenced (none in current implementation)


def rewrite_code(code):
  rewriter = TestRewriter()
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_kaiming_uniform():
  """
  Input: nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
  Output: nn.initializers.he_uniform(a=math.sqrt(5))
  """
  code = "nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))"
  res = rewrite_code(code)

  assert "nn.initializers.he_uniform" in res
  assert "self.weight" not in res  # Arg 0 stripped
  assert "a=math.sqrt(5)" in res  # Kwargs preserved


def test_constant_fill():
  """
  Input: nn.init.constant_(self.bias, 0)
  Output: nn.initializers.constant(0)
  """
  code = "nn.init.constant_(self.bias, 0)"
  res = rewrite_code(code)

  assert "nn.initializers.constant(0)" in res
  assert "self.bias" not in res


def test_xavier_normal():
  """
  Input: torch.nn.init.xavier_normal_(w)
  Output: nn.initializers.glorot_normal()
  """
  code = "torch.nn.init.xavier_normal_(w)"
  res = rewrite_code(code)

  assert "nn.initializers.glorot_normal()" in res


def test_ignore_unknown_init():
  """
  Input: nn.init.unknown_method_(w)
  Output: Unchanged (safety fallback)
  """
  code = "nn.init.custom_(w)"
  res = rewrite_code(code)

  # mixin logic returns node unchanged if not map
  assert "nn.init.custom_" in res
  assert "w" in res
