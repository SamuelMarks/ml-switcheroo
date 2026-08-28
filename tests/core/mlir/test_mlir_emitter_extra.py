"""Test suite for the Mlir Emitter Extra module."""

import typing
import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.cst import ModuleNode


def convert_code(code: str) -> str:
  """Converts code."""
  tree: cst.Module = cst.parse_module(code.strip())
  emitter = PythonToMlirEmitter()
  mlir_node: ModuleNode = emitter.convert(tree)
  return mlir_node.to_text()


def test_module_header_newline_and_trivia() -> None:
  """Verifies the behavior of module header newline and trivia."""
  code: str = "\n# A leading comment\n\nclass A:\n    pass\n"
  mlir: str = convert_code(code)
  assert "sw.module" in mlir
  assert "// A leading comment" in mlir


def test_statement_leading_trivia() -> None:
  """Verifies the behavior of statement leading trivia."""
  code: str = "\ndef func(a):\n\n    # Statement leading comment\n    return a\n"
  mlir: str = convert_code(code)
  assert "// Statement leading comment" in mlir


def test_expr_statement_and_func_call() -> None:
  """Verifies the behavior of expr statement and function call."""
  code: str = "\ndef func(a):\n    print(a)\n    return a\n"
  mlir: str = convert_code(code)
  assert "sw.op" in mlir
  assert "print" in mlir


def test_imports() -> None:
  """Verifies the behavior of imports."""
  code: str = "\nimport numpy as np\nimport os\nfrom math import pi as p, sqrt\nfrom some_module import *\n"
  mlir: str = convert_code(code)
  assert "sw.import" in mlir


def test_class_inheritance() -> None:
  """Verifies the behavior of class inheritance."""
  code: str = "\nclass MyLayer(nn.Module, Base):\n    pass\n"
  mlir: str = convert_code(code)
  assert 'bases = ["nn.Module", "Base"]' in mlir


def test_attribute_assignment_unresolved() -> None:
  """Verifies the behavior of attribute assignment unresolved."""
  code: str = "\ndef __init__(self):\n    unresolved.layer1 = 10\n"
  mlir: str = convert_code(code)
  assert "sw.setattr" not in mlir


def test_attribute_assignment() -> None:
  """Verifies the behavior of attribute assignment."""
  code: str = "\ndef __init__(self):\n    self.layer1 = 10\n"
  mlir: str = convert_code(code)
  assert "sw.setattr" in mlir


def test_flatten_attr_none() -> None:
  """Verifies the behavior of flatten attribute none."""
  code: str = "\ndef func(a):\n    return a().attr\n"
  mlir: str = convert_code(code)
  assert "sw.return" in mlir


def test_all_binops() -> None:
  """Verifies the behavior of all binops."""
  code: str = "\ndef math_ops(a, b):\n    v1 = a - b\n    v2 = a // b\n    v3 = a % b\n    v4 = a ** b\n    v5 = a @ b\n    v6 = a << b\n    v7 = a >> b\n    v8 = a & b\n    v9 = a | b\n    v10 = a ^ b\n    return v10\n"
  mlir: str = convert_code(code)
  assert "binop.sub" in mlir


def test_unknown_binop() -> None:
  """Verifies the behavior of unknown binop."""

  class DummyBinOp(cst.BaseBinaryOp):
    """Dummy Bin Op class for testing purposes."""

    def _visit_and_replace_children(self, visitor: typing.Any) -> typing.Any:
      """Mock implementation of  visit and replace children."""
      return self

    def _codegen_impl(self, state: typing.Any, default: typing.Any) -> None:
      """Mock implementation of  codegen impl."""
      pass

  emitter = PythonToMlirEmitter()
  assert emitter._get_binop_str(DummyBinOp()) == "unknown"


def test_kwargs_in_call() -> None:
  """Verifies the behavior of keyword arguments in call."""
  code: str = "\ndef forward(x):\n    return torch.nn.functional.relu(x, inplace=True)\n"
  mlir: str = convert_code(code)
  assert "arg_keywords" in mlir


def test_call_local_variable() -> None:
  """Verifies the behavior of call local variable."""
  code: str = "\ndef apply_func(func, x):\n    return func(x)\n"
  mlir: str = convert_code(code)
  assert "sw.call" in mlir


def test_unhandled_expression() -> None:
  """Verifies the behavior of unhandled expression."""
  code: str = "\ndef func():\n    return lambda y: y\n"
  mlir: str = convert_code(code)
  assert "%error" in mlir


def test_complex_type_annotation() -> None:
  """Verifies the behavior of complex type annotation."""
  code: str = "\ndef f(x: torch.Tensor, y: List[int]):\n    pass\n"
  mlir: str = convert_code(code)
  assert '!sw.type<"torch.Tensor">' in mlir


def test_flatten_attr_none_cases() -> None:
  """Verifies the behavior of flatten attribute none cases."""
  emitter = PythonToMlirEmitter()
  code: str = "\nclass A(b()):\n    pass\n"
  emitter.convert(cst.parse_module(code.strip()))
  code = "\ndef f():\n    b().attr = 1\n"
  emitter.convert(cst.parse_module(code.strip()))
  code = "\ndef f():\n    b().attr()\n"
  emitter.convert(cst.parse_module(code.strip()))


def test_extract_trivia_newlines() -> None:
  """Extracts trivia newlines."""
  emitter = PythonToMlirEmitter()
  node = cst.SimpleStatementLine(
    body=[cst.Pass()], leading_lines=[cst.EmptyLine(indent=False, comment=None, newline=cst.Newline(value="\n"))]
  )
  trivia: list[typing.Any] = emitter._extract_trivia(node)
  assert len(trivia) == 1
  assert trivia[0].text == "\n"
