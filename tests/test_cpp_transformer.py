"""Test module."""

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
  ReturnStatement,
  TypeIdentifier,
  VariableDeclaration,
)
from ml_switcheroo.core.compiler.backends.cpp.transformer import CppCSTTransformer


def test_transformer_generic_visit_cppmodule() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: CppModule = CppModule(includes=[IncludeDirective(path="iostream")], body=[ReturnStatement()])
  result: CppNode = transformer.visit(node)
  assert isinstance(result, CppModule)


def test_transformer_function_definition() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: FunctionDefinition = FunctionDefinition(
    return_type=TypeIdentifier(name="int"),
    name="foo",
    arguments=[FunctionArgument(type_id=TypeIdentifier(name="int"), name="x")],
    body=[ReturnStatement(value=Identifier(name="x"))],
  )
  result: CppNode = transformer.visit(node)
  assert isinstance(result, FunctionDefinition)


def test_transformer_variable_declaration() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: VariableDeclaration = VariableDeclaration(
    type_id=TypeIdentifier(name="int"), name="x", initializer=Identifier(name="y")
  )
  result: CppNode = transformer.visit(node)
  assert isinstance(result, VariableDeclaration)

  node_no_init: VariableDeclaration = VariableDeclaration(type_id=TypeIdentifier(name="int"), name="x")
  result_no_init: CppNode = transformer.visit(node_no_init)
  assert isinstance(result_no_init, VariableDeclaration)

  node_str_init: VariableDeclaration = VariableDeclaration(
    type_id=TypeIdentifier(name="int"), name="x", initializer=Identifier(name="1")
  )
  result_str_init: CppNode = transformer.visit(node_str_init)
  assert isinstance(result_str_init, VariableDeclaration)


def test_transformer_binary_expression() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: BinaryExpression = BinaryExpression(left=Identifier(name="a"), operator="+", right=Identifier(name="b"))
  result: CppNode = transformer.visit(node)
  assert isinstance(result, BinaryExpression)


def test_transformer_method_call() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: MethodCall = MethodCall(name="foo", arguments=[Identifier(name="a")])
  result: CppNode = transformer.visit(node)
  assert isinstance(result, MethodCall)


def test_transformer_block_statement() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: BlockStatement = BlockStatement(statements=[ReturnStatement()])
  result: CppNode = transformer.visit(node)
  assert isinstance(result, BlockStatement)


def test_transformer_pybind_module() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node: PyBindModule = PyBindModule(
    name="mod", module_var="m", defs=[PyBindDef(name="f", function_ref="f_impl", docstring="doc")]
  )
  result: CppNode = transformer.visit(node)
  assert isinstance(result, PyBindModule)


def test_transformer_identifiers() -> None:
  """Docstring."""
  transformer: CppCSTTransformer = CppCSTTransformer()
  node_id: Identifier = Identifier(name="x")
  assert transformer.visit(node_id) is node_id

  node_type: TypeIdentifier = TypeIdentifier(name="int")
  assert transformer.visit(node_type) is node_type


def test_transformer_custom_node() -> None:
  """Docstring."""

  class DummyNode(CppNode):
    """Docstring."""

    def to_text(self) -> str:
      """Docstring."""
      return ""

  transformer: CppCSTTransformer = CppCSTTransformer()
  node: DummyNode = DummyNode()
  result: CppNode = transformer.visit(node)
  assert result is node
