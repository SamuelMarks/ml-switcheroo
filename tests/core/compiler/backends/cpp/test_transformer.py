"""Unit tests for `CppCSTTransformer` (`ml_switcheroo.core.compiler.backends.cpp.transformer`)."""

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


class _IdentifierRenamer(CppCSTTransformer):
  """Rename identifiers matching `src` to `dst`."""

  def __init__(self, src: str, dst: str) -> None:
    """Store source and destination identifier names.

    Args:
        src: Source identifier name.
        dst: Replacement identifier name.
    """
    self.src = src
    self.dst = dst

  def visit_Identifier(self, node: Identifier) -> Identifier:
    """Rename matching `Identifier` node in place.

    Args:
        node: Visited `Identifier` node.

    Returns:
        Mutated or unchanged `Identifier` node.
    """
    if node.name == self.src:
      node.name = self.dst
    return node


def test_cpp_cst_transformer_traversal_and_mutation() -> None:
  """Verify `CppCSTTransformer` traverses all C++ CST node variants and subclasses mutate identifiers."""
  base = CppCSTTransformer()
  id_node = Identifier("x")
  type_node = TypeIdentifier("int")
  assert base.visit(id_node) is id_node
  assert base.visit(type_node) is type_node

  assert base.visit(VariableDeclaration(TypeIdentifier("int"), "x")).to_text() == "int x;"
  assert base.visit(VariableDeclaration(TypeIdentifier("int"), "y", "0")).to_text() == "int y = 0;"
  assert base.visit(ReturnStatement()).to_text() == "return;"

  import pytest

  from ml_switcheroo.exceptions import DialectLoweringError

  with pytest.raises(DialectLoweringError, match="Expected TypeIdentifier"):
    base._visit_typed(Identifier("x"), TypeIdentifier)

  class _CustomNode(CppNode):
    """Custom `CppNode` without dedicated `visit_*` method."""

    def to_text(self) -> str:
      """Return empty string.

      Returns:
          Empty string `""`.
      """
      return ""

  custom = _CustomNode()
  assert base.visit(custom) is custom

  mod = CppModule(
    includes=[IncludeDirective("iostream")],
    body=[
      FunctionDefinition(
        return_type=TypeIdentifier("int"),
        name="my_func",
        arguments=[FunctionArgument(TypeIdentifier("int"), "a")],
        body=[
          VariableDeclaration(TypeIdentifier("int"), "b", BinaryExpression(Identifier("a"), "+", Identifier("w"))),
          VariableDeclaration(TypeIdentifier("int"), "c", MethodCall("add", [Identifier("a")])),
          BlockStatement([RawStatement("a++;")]),
          ReturnStatement(Identifier("b")),
        ],
      ),
      PyBindModule("name", "m", [PyBindDef("my_func", "my_func", "doc")]),
    ],
  )
  assert base.visit(mod) is mod

  renamer = _IdentifierRenamer("a", "A")
  transformed = renamer.visit(mod)
  assert isinstance(transformed, CppModule)
  fn_out = transformed.body[0]
  assert isinstance(fn_out, FunctionDefinition)
  assert fn_out.arguments[0].name == "a"
  assert "int b = A + w;" in fn_out.to_text()
  assert "int c = add(A);" in fn_out.to_text()
