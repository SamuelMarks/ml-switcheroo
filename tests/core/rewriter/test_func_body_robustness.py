"""
Tests for Robust Super Init Detection.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.func_body import FuncBodyMixin


class MockRewriter(FuncBodyMixin):
  pass


@pytest.fixture
def rewriter():
  return MockRewriter()


def test_detect_assignment_super(rewriter):
  """
  Scenario: _1 = super().__init__()
  Expect: Detected as super init.
  """
  code = "_1 = super().__init__()"
  stmt = cst.parse_module(code).body[0]
  assert rewriter._is_super_init_stmt(stmt) is True


def test_detect_expression_super(rewriter):
  """
  Scenario: super().__init__()
  Expect: Detected as super init.
  """
  code = "super().__init__()"
  stmt = cst.parse_module(code).body[0]
  assert rewriter._is_super_init_stmt(stmt) is True


def test_detect_args_super(rewriter):
  """
  Scenario: super(MyClass, self).__init__()
  Expect: Detected as super init.
  """
  code = "super(MyClass, self).__init__()"
  stmt = cst.parse_module(code).body[0]
  assert rewriter._is_super_init_stmt(stmt) is True


def test_has_super_init_check(rewriter):
  """
  Verify that _has_super_init finds the assignment version deep in body.
  """
  code = """ 
def __init__(self): 
    x = 1
    _0 = super().__init__() 
"""
  func = cst.parse_module(code).body[0]
  assert rewriter._has_super_init(func) is True


def test_ensure_super_init_idempotent_with_assignment(rewriter):
  """
  Verify that _ensure_super_init does NOT inject a duplicate if assignment version exists.
  """
  code = """ 
def __init__(self): 
    _ = super().__init__() 
"""
  func = cst.parse_module(code).body[0]
  new_func = rewriter._ensure_super_init(func)

  source = cst.Module(body=[new_func]).code
  # Should only have one super init line
  assert source.count("super") == 1


def test_strip_super_init_assignment(rewriter):
  """
  Verify _strip_super_init removes the assignment version.
  """
  code = """ 
def setup(self): 
    _ = super().__init__() 
    self.x = 1
"""
  func = cst.parse_module(code).body[0]
  new_func = rewriter._strip_super_init(func)

  source = cst.Module(body=[new_func]).code
  assert "super" not in source
  assert "self.x = 1" in source
