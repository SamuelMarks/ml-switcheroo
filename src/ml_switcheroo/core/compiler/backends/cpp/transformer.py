"""C++ CST Transformer.

Provides a visitor pattern for traversing and mutating C++ CST nodes.
"""

from typing import cast
from ml_switcheroo.core.compiler.backends.cpp.cst import (
  CppNode,
  CppModule,
  IncludeDirective,
  FunctionDefinition,
  FunctionArgument,
  TypeIdentifier,
  VariableDeclaration,
  ReturnStatement,
  Identifier,
  Expression,
  BinaryExpression,
  MethodCall,
  BlockStatement,
  PyBindModule,
  PyBindDef,
)


class CppCSTTransformer:
  """Base visitor/transformer for CppNode trees."""

  def visit(self, node: CppNode) -> CppNode:
    """Visits a node and dispatches to the correct visit_* method.

    Args:
        node (CppNode): The concrete syntax tree node to visit.

    Returns:
        CppNode: The visited/transformed concrete syntax tree node.

    """
    method_name = f"visit_{node.__class__.__name__}"
    visitor = getattr(self, method_name, self.generic_visit)
    return visitor(node)

  def generic_visit(self, node: CppNode) -> CppNode:
    """Fallback visitor that traverses children.

    Args:
        node (CppNode): The concrete syntax tree node whose children will be visited.

    Returns:
        CppNode: The visited/transformed node after child traversal.

    """
    if isinstance(node, CppModule):
      node.includes = [cast(IncludeDirective, self.visit(inc)) for inc in node.includes]
      node.body = [self.visit(stmt) for stmt in node.body]
    elif isinstance(node, FunctionDefinition):
      node.return_type = cast(TypeIdentifier, self.visit(node.return_type))
      node.arguments = [cast(FunctionArgument, self.visit(arg)) for arg in node.arguments]
      node.body = [self.visit(stmt) for stmt in node.body]
    elif isinstance(node, FunctionArgument):
      node.type_id = cast(TypeIdentifier, self.visit(node.type_id))
    elif isinstance(node, VariableDeclaration):
      node.type_id = cast(TypeIdentifier, self.visit(node.type_id))
      if isinstance(node.initializer, CppNode):
        node.initializer = cast(Expression, self.visit(node.initializer))
    elif isinstance(node, ReturnStatement):
      if node.value:
        node.value = cast(Expression, self.visit(node.value))
    elif isinstance(node, BinaryExpression):
      node.left = cast(Expression, self.visit(node.left))
      node.right = cast(Expression, self.visit(node.right))
    elif isinstance(node, MethodCall):
      node.arguments = [cast(Expression, self.visit(arg)) for arg in node.arguments]
    elif isinstance(node, BlockStatement):
      node.statements = [self.visit(stmt) for stmt in node.statements]
    elif isinstance(node, PyBindModule):
      node.defs = [cast(PyBindDef, self.visit(d)) for d in node.defs]

    return node

  def visit_CppModule(self, node: CppModule) -> CppNode:
    """Visit a CppModule node.

    Args:
        node (CppModule): The CppModule node to visit.

    Returns:
        CppNode: The visited/transformed CppModule node.

    """
    return self.generic_visit(node)

  def visit_FunctionDefinition(self, node: FunctionDefinition) -> CppNode:
    """Visit a FunctionDefinition node.

    Args:
        node (FunctionDefinition): The FunctionDefinition node to visit.

    Returns:
        CppNode: The visited/transformed FunctionDefinition node.

    """
    return self.generic_visit(node)

  def visit_VariableDeclaration(self, node: VariableDeclaration) -> CppNode:
    """Visit a VariableDeclaration node.

    Args:
        node (VariableDeclaration): The VariableDeclaration node to visit.

    Returns:
        CppNode: The visited/transformed VariableDeclaration node.

    """
    return self.generic_visit(node)

  def visit_ReturnStatement(self, node: ReturnStatement) -> CppNode:
    """Visit a ReturnStatement node.

    Args:
        node (ReturnStatement): The ReturnStatement node to visit.

    Returns:
        CppNode: The visited/transformed ReturnStatement node.

    """
    return self.generic_visit(node)

  def visit_BinaryExpression(self, node: BinaryExpression) -> CppNode:
    """Visit a BinaryExpression node.

    Args:
        node (BinaryExpression): The BinaryExpression node to visit.

    Returns:
        CppNode: The visited/transformed BinaryExpression node.

    """
    return self.generic_visit(node)

  def visit_MethodCall(self, node: MethodCall) -> CppNode:
    """Visit a MethodCall node.

    Args:
        node (MethodCall): The MethodCall node to visit.

    Returns:
        CppNode: The visited/transformed MethodCall node.

    """
    return self.generic_visit(node)

  def visit_Identifier(self, node: Identifier) -> CppNode:
    """Visit an Identifier node.

    Args:
        node (Identifier): The Identifier node to visit.

    Returns:
        CppNode: The same Identifier node (leaf node).

    """
    return node

  def visit_TypeIdentifier(self, node: TypeIdentifier) -> CppNode:
    """Visit a TypeIdentifier node.

    Args:
        node (TypeIdentifier): The TypeIdentifier node to visit.

    Returns:
        CppNode: The same TypeIdentifier node (leaf node).

    """
    return node
