"""Unit tests for the C++ CST transformer in the compiler backend.

This module contains tests verifying that the CppCSTTransformer and its subclasses
can correctly traverse and transform C++ Concrete Syntax Tree (CST) nodes.
"""

from ml_switcheroo.core.compiler.backends.cpp.cst import Identifier, BinaryExpression
from ml_switcheroo.core.compiler.backends.cpp.transformer import CppCSTTransformer


class Renamer(CppCSTTransformer):
  """A helper C++ CST transformer that renames identifiers named 'input' to 'x'.

  This class subclasses CppCSTTransformer to demonstrate and test node visitor
  behavior during tree transformation.
  """

  def visit_Identifier(self, node: Identifier) -> Identifier:
    """Visits and transforms an Identifier node.

    If the identifier's name is 'input', it is renamed to 'x'.

    Args:
        node (Identifier): The C++ CST Identifier node to visit and potentially transform.

    Returns:
        Identifier: The visited and potentially modified Identifier node.
    """
    if node.name == "input":
      node.name = "x"
    return node


def test_transformer() -> None:
  """Verifies that the Renamer transformer correctly replaces identifier names in a binary expression.

  Args:
      None

  Returns:
      None
  """
  expr = BinaryExpression(Identifier("input"), "*", Identifier("weights"))
  t = Renamer()
  expr2 = t.visit(expr)
  assert expr2.left.name == "x"
  assert expr2.right.name == "weights"
  assert expr2.to_text() == "x * weights"


def test_transformer_variable_decl() -> None:
  """Tests VariableDeclaration transformation with a non-node string initializer.

  Args:
      None

  Returns:
      None
  """
  from ml_switcheroo.core.compiler.backends.cpp.cst import VariableDeclaration, TypeIdentifier

  decl = VariableDeclaration(TypeIdentifier("int"), "y", "0")
  t = Renamer()
  decl2 = t.visit(decl)
  assert decl2.to_text() == "int y = 0;"


def test_transformer_return_empty() -> None:
  """Tests ReturnStatement transformation when there is no return value.

  Args:
      None

  Returns:
      None
  """
  from ml_switcheroo.core.compiler.backends.cpp.cst import ReturnStatement

  ret = ReturnStatement()
  t = Renamer()
  ret2 = t.visit(ret)
  assert ret2.to_text() == "return;"
