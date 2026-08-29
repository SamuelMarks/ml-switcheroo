"""Unit tests for the C++ CST transformer in the compiler backend.

This module contains tests verifying that the CppCSTTransformer and its subclasses
can correctly traverse and transform C++ Concrete Syntax Tree (CST) nodes.
"""

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  BinaryExpression,
  BlockStatement,
  CppModule,
  CppNode,
  FunctionArgument,
  FunctionDefinition,
  Identifier,
  IncludeDirective,
  MethodCall,
  PyBindDef,
  PyBindModule,
  RawStatement,
  ReturnStatement,
  TypeIdentifier,
  VariableDeclaration,
)
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
  from ml_switcheroo.core.compiler.backends.cpp.cst import TypeIdentifier, VariableDeclaration

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


# --- Merged from test_transformer_extra.py ---

"""Unit tests for validating the traversal and transformation capabilities of the C++ Concrete Syntax Tree (CST) Transformer.

This module contains extra test scenarios targeting the CppCSTTransformer class, ensuring
that the default visitor behavior correctly propagates across all major C++ AST node types
and can be subclassed to target specific nodes.
"""


def test_full_transform() -> None:
  """Test the CppCSTTransformer with a complete C++ module CST.

  This test constructs a complex `CppModule` node featuring functions, variable
  declarations, method calls, block statements, and PyBind definitions. It verifies
  that the base transformer visits all nodes and returns the original node structure
  unmodified, while a subclassed transformer (`Mover`) successfully targets and
  mutates specific `Identifier` nodes during traversal.

  Args:
      None

  Returns:
      None
  """
  t = CppCSTTransformer()
  mod = CppModule(
    includes=[IncludeDirective("iostream")],
    body=[
      FunctionDefinition(
        return_type=TypeIdentifier("int"),
        name="my_func",
        arguments=[FunctionArgument(TypeIdentifier("int"), "a")],
        body=[
          VariableDeclaration(TypeIdentifier("int"), "b", BinaryExpression(Identifier("a"), "+", Identifier("a"))),
          VariableDeclaration(TypeIdentifier("int"), "c", MethodCall("add", [Identifier("a")])),
          BlockStatement([RawStatement("a++;")]),
          ReturnStatement(Identifier("b")),
        ],
      ),
      PyBindModule("name", "m", [PyBindDef("my_func", "my_func", "doc")]),
    ],
  )
  # just traversing should not crash and should return the node itself
  res: CppNode = t.visit(mod)
  assert res is mod
  assert getattr(t.visit(Identifier("foo")), "name", None) == "foo"
  assert getattr(t.visit(TypeIdentifier("int")), "name", None) == "int"

  class Mover(CppCSTTransformer):
    """A specialized AST transformer for mutating identifier names during C++ CST traversal.

    This subclass overrides visit methods to perform target-specific transformations on the CST.
    """

    def visit_Identifier(self, node: Identifier) -> Identifier:
      """Visit an Identifier node and rename it if it matches specific criteria.

      If the identifier name is 'a', it is renamed to 'A'.

      Args:
          node (Identifier): The CST Identifier node being visited.

      Returns:
          Identifier: The visited (and potentially modified) Identifier node.
      """
      if node.name == "a":
        node.name = "A"
      return node

  m = Mover()
  res2: CppNode = m.visit(mod)
  assert (
    getattr(res2, "body")[0].arguments[0].name == "a"
  )  # not changed because it's a FunctionArgument string not Identifier
