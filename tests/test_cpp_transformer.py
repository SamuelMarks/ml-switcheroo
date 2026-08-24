"""Test module."""

from ml_switcheroo.core.compiler.backends.cpp.transformer import CppCSTTransformer
from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppModule,
  IncludeDirective,
  FunctionDefinition,
  FunctionArgument,
  TypeIdentifier,
  VariableDeclaration,
  ReturnStatement,
  BinaryExpression,
  MethodCall,
  BlockStatement,
  PyBindModule,
  PyBindDef,
  Identifier,
  CppNode,
)


def test_transformer_generic_visit_cppmodule():
  """Test element."""
  transformer = CppCSTTransformer()
  node = CppModule(includes=[IncludeDirective(path="iostream")], body=[ReturnStatement()])
  result = transformer.visit(node)
  assert isinstance(result, CppModule)


def test_transformer_function_definition():
  """Test element."""
  transformer = CppCSTTransformer()
  node = FunctionDefinition(
    return_type=TypeIdentifier("int"),
    name="foo",
    arguments=[FunctionArgument(TypeIdentifier("int"), "x")],
    body=[ReturnStatement(Identifier("x"))],
  )
  result = transformer.visit(node)
  assert isinstance(result, FunctionDefinition)


def test_transformer_variable_declaration():
  """Test element."""
  transformer = CppCSTTransformer()
  node = VariableDeclaration(type_id=TypeIdentifier("int"), name="x", initializer=Identifier("y"))
  result = transformer.visit(node)
  assert isinstance(result, VariableDeclaration)

  node_no_init = VariableDeclaration(type_id=TypeIdentifier("int"), name="x")
  result_no_init = transformer.visit(node_no_init)
  assert isinstance(result_no_init, VariableDeclaration)

  node_str_init = VariableDeclaration(type_id=TypeIdentifier("int"), name="x", initializer="1")
  result_str_init = transformer.visit(node_str_init)
  assert isinstance(result_str_init, VariableDeclaration)


def test_transformer_binary_expression():
  """Test element."""
  transformer = CppCSTTransformer()
  node = BinaryExpression(left=Identifier("a"), operator="+", right=Identifier("b"))
  result = transformer.visit(node)
  assert isinstance(result, BinaryExpression)


def test_transformer_method_call():
  """Test element."""
  transformer = CppCSTTransformer()
  node = MethodCall(name="foo", arguments=[Identifier("a")])
  result = transformer.visit(node)
  assert isinstance(result, MethodCall)


def test_transformer_block_statement():
  """Test element."""
  transformer = CppCSTTransformer()
  node = BlockStatement(statements=[ReturnStatement()])
  result = transformer.visit(node)
  assert isinstance(result, BlockStatement)


def test_transformer_pybind_module():
  """Test element."""
  transformer = CppCSTTransformer()
  node = PyBindModule(name="mod", module_var="m", defs=[PyBindDef(name="f", function_ref="f_impl", docstring="doc")])
  result = transformer.visit(node)
  assert isinstance(result, PyBindModule)


def test_transformer_identifiers():
  """Test element."""
  transformer = CppCSTTransformer()
  node_id = Identifier("x")
  assert transformer.visit(node_id) is node_id

  node_type = TypeIdentifier("int")
  assert transformer.visit(node_type) is node_type


def test_transformer_custom_node():
  """Test element."""

  class DummyNode(CppNode):
    def to_text(self):
      return ""

  transformer = CppCSTTransformer()
  node = DummyNode()
  result = transformer.visit(node)
  assert result is node
