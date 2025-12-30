"""
Tests for PythonToMlirEmitter.

Verifies:
1.  Variable scoping and SSA ID generation.
2.  Class & Function definitions mapping to sw.module/sw.func.
3.  Comment/Trivia preservation.
4.  Recursive expression evaluation (calls, attributes).
5.  Binary Operations (math expressions).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter


def convert_code(code: str):
  """Helper to run emitter and get text."""
  tree = cst.parse_module(code.strip())
  emitter = PythonToMlirEmitter()
  mlir_node = emitter.convert(tree)
  return mlir_node.to_text()


def test_class_definition():
  """
  Input: class MyNet: pass
  Output: sw.module {sym_name = "MyNet"} { ... }
  """
  code = "class MyNet:\n    pass"
  mlir = convert_code(code)

  assert "sw.module" in mlir
  assert 'sym_name = "MyNet"' in mlir


def test_function_definition_with_args():
  """
  Input:
      def forward(self, x):
          return x

  Output:
      sw.func {sym_name = "forward"} {
      ^entry(%self0: !sw.unknown, %x1: !sw.unknown):
          sw.return (%x1)
      }
  """
  code = """ 
def forward(self, x): 
    return x
"""
  mlir = convert_code(code)

  assert "sw.func" in mlir
  assert 'sym_name = "forward"' in mlir
  assert "^entry(%self0: !sw.unknown, %x1: !sw.unknown):" in mlir
  assert "sw.return (%x1)" in mlir


def test_attribute_call_mapping():
  """
  Input:
      def f(self, x):
          return self.layer(x)

  Logic:
      1. %0 = sw.getattr %self "layer"
      2. %1 = sw.call %0 (%x)
      3. sw.return %1
  """
  code = """ 
def f(self, x): 
    return self.layer(x) 
"""
  mlir = convert_code(code)

  assert "sw.getattr" in mlir
  assert 'name = "layer"' in mlir
  assert "sw.call" in mlir
  assert "sw.return" in mlir


def test_constructor_op_mapping():
  """
  Input:
      x = torch.nn.Conv2d(1, 32)

  Logic:
      1. Constants 1, 32
      2. sw.op { type="torch.nn.Conv2d" }
  """
  code = "x = torch.nn.Conv2d(1, 32)"
  mlir = convert_code(code)

  assert "sw.constant" in mlir
  assert "value = 1" in mlir
  assert "value = 32" in mlir
  assert "sw.op" in mlir
  assert 'type = "torch.nn.Conv2d"' in mlir


def test_trivia_preservation():
  """
  Input:
      # My Comment
      class A: pass

  Output: Contains // My Comment
  """
  code = """ 
# My Comment
class A: 
    pass
"""
  mlir = convert_code(code)
  assert "// My Comment" in mlir


def test_scope_isolation():
  """
  Verify variables vars in different functions get unique SSA IDs
  and don't conflict logic.
  """
  code = """ 
def f1(a): 
    return a
def f2(a): 
    return a
"""
  mlir = convert_code(code)

  # f1
  assert 'sym_name = "f1"' in mlir
  # f2 exists
  assert 'sym_name = "f2"' in mlir
  # Both should have valid return ops
  assert mlir.count("sw.return") == 2


def test_typed_args():
  """
  Input: def f(x: int): pass
  Output: type should be !sw.type<"int">
  """
  code = "def f(x: int): pass"
  mlir = convert_code(code)

  assert '!sw.type<"int">' in mlir


def test_binary_math_expression():
  """
  Input: x = 32 * 26 * 26
  Logic:
      1. Constants for 32, 26, 26
      2. sw.op(..., ...) {type="binop.mul"} recursed
  """
  code = "x = 32 * 26 * 26"
  mlir = convert_code(code)

  assert "sw.op" in mlir
  assert 'type = "binop.mul"' in mlir
  # Ensure it appears at least twice (32*26) and (result * 26)
  assert mlir.count('type = "binop.mul"') == 2


def test_mixed_binary_math():
  """
  Input: y = a + b / 2
  Logic:
      1. binop.div
      2. binop.add
  """
  code = "y = a + b / 2"
  mlir = convert_code(code)

  assert 'type = "binop.div"' in mlir
  assert 'type = "binop.add"' in mlir
  # Check standard lookup fallback for a and b
  assert "@a" in mlir
  assert "@b" in mlir
