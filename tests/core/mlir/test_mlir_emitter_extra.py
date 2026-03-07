"""Module docstring."""

import pytest
import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter


def convert_code(code: str):
  """Function docstring."""
  tree = cst.parse_module(code.strip())
  emitter = PythonToMlirEmitter()
  mlir_node = emitter.convert(tree)
  return mlir_node.to_text()


def test_module_header_newline_and_trivia():
  """Function docstring."""
  code = """
# A leading comment

class A:
    pass
"""
  mlir = convert_code(code)
  assert "sw.module" in mlir
  assert "// A leading comment" in mlir


def test_statement_leading_trivia():
  """Function docstring."""
  code = """
def func(a):
    
    # Statement leading comment
    return a
"""
  mlir = convert_code(code)
  assert "// Statement leading comment" in mlir


def test_expr_statement_and_func_call():
  """Function docstring."""
  code = """
def func(a):
    print(a)
    return a
"""
  mlir = convert_code(code)
  assert "sw.op" in mlir
  assert "print" in mlir


def test_imports():
  """Function docstring."""
  code = """
import numpy as np
import os
from math import pi as p, sqrt
from some_module import *
"""
  mlir = convert_code(code)
  assert "sw.import" in mlir


def test_class_inheritance():
  """Function docstring."""
  code = """
class MyLayer(nn.Module, Base):
    pass
"""
  mlir = convert_code(code)
  assert 'bases = ["nn.Module", "Base"]' in mlir


def test_attribute_assignment_unresolved():
  """Function docstring."""
  code = """
def __init__(self):
    unresolved.layer1 = 10
"""
  mlir = convert_code(code)
  assert "sw.setattr" not in mlir


def test_attribute_assignment():
  """Function docstring."""
  code = """
def __init__(self):
    self.layer1 = 10
"""
  mlir = convert_code(code)
  assert "sw.setattr" in mlir


def test_flatten_attr_none():
  """Function docstring."""
  code = """
def func(a):
    return a().attr
"""
  mlir = convert_code(code)
  # This invokes flatten_attr on a Call, which returns None.
  # We just want to ensure it doesn't crash and covers the None branch.
  assert "sw.return" in mlir


def test_all_binops():
  """Function docstring."""
  code = """
def math_ops(a, b):
    v1 = a - b
    v2 = a // b
    v3 = a % b
    v4 = a ** b
    v5 = a @ b
    v6 = a << b
    v7 = a >> b
    v8 = a & b
    v9 = a | b
    v10 = a ^ b
    return v10
"""
  mlir = convert_code(code)
  assert "binop.sub" in mlir


def test_unknown_binop():
  """Function docstring."""

  # To hit the 'unknown' branch, we'll manually call _get_binop_str
  # with an empty node
  class DummyBinOp(cst.BaseBinaryOp):
    """Class docstring."""

    def _visit_and_replace_children(self, visitor):
      """Function docstring."""
      return self

    def _codegen_impl(self, state, default):
      """Function docstring."""
      pass

  emitter = PythonToMlirEmitter()
  assert emitter._get_binop_str(DummyBinOp()) == "unknown"


def test_kwargs_in_call():
  """Function docstring."""
  code = """
def forward(x):
    return torch.nn.functional.relu(x, inplace=True)
"""
  mlir = convert_code(code)
  assert "arg_keywords" in mlir


def test_call_local_variable():
  """Function docstring."""
  code = """
def apply_func(func, x):
    return func(x)
"""
  mlir = convert_code(code)
  assert "sw.call" in mlir


def test_unhandled_expression():
  """Function docstring."""
  code = """
def func():
    return lambda y: y
"""
  mlir = convert_code(code)
  assert "%error" in mlir


def test_complex_type_annotation():
  """Function docstring."""
  code = """
def f(x: torch.Tensor, y: List[int]):
    pass
"""
  mlir = convert_code(code)
  assert '!sw.type<"torch.Tensor">' in mlir


def test_flatten_attr_none_cases():
  """Function docstring."""
  emitter = PythonToMlirEmitter()

  # 1. Base class is a Call
  code = """
class A(b()):
    pass
"""
  emitter.convert(cst.parse_module(code.strip()))

  # 2. Assign to an attribute of a call
  code = """
def f():
    b().attr = 1
"""
  emitter.convert(cst.parse_module(code.strip()))

  # 3. Call an attribute of a call
  code = """
def f():
    b().attr()
"""
  emitter.convert(cst.parse_module(code.strip()))


def test_extract_trivia_newlines():
  """Function docstring."""
  emitter = PythonToMlirEmitter()
  # We can craft an EmptyLine with a Newline object directly
  node = cst.SimpleStatementLine(
    body=[cst.Pass()], leading_lines=[cst.EmptyLine(indent=False, comment=None, newline=cst.Newline(value="\n"))]
  )
  trivia = emitter._extract_trivia(node)
  assert len(trivia) == 1
  assert trivia[0].content == "\n"
