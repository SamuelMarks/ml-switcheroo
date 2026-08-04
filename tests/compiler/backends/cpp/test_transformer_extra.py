"""Unit tests for validating the traversal and transformation capabilities of the C++ Concrete Syntax Tree (CST) Transformer.

This module contains extra test scenarios targeting the CppCSTTransformer class, ensuring
that the default visitor behavior correctly propagates across all major C++ AST node types
and can be subclassed to target specific nodes.
"""

from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppModule,
  IncludeDirective,
  FunctionDefinition,
  FunctionArgument,
  TypeIdentifier,
  VariableDeclaration,
  ReturnStatement,
  Identifier,
  BinaryExpression,
  MethodCall,
  RawStatement,
  BlockStatement,
  PyBindModule,
  PyBindDef,
)
from ml_switcheroo.core.compiler.backends.cpp.transformer import CppCSTTransformer


def test_full_transform():
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
  res = t.visit(mod)
  assert res is mod
  assert t.visit(Identifier("foo")).name == "foo"
  assert t.visit(TypeIdentifier("int")).name == "int"

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
  res2 = m.visit(mod)
  assert res2.body[0].arguments[0].name == "a"  # not changed because it's a FunctionArgument string not Identifier
