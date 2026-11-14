"""C++ Concrete Syntax Tree (CST) Builder API.

Provides a structural way to generate C++ code, avoiding brittle string
concatenation and macros. Inspired by `libcst` for Python.
"""

import typing

import pydantic

from ml_switcheroo.core.cst.base import CstFieldValue, CSTNode, TriviaInput


class CppNode(CSTNode):
  """Base class for all C++ CST nodes."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize CppNode.

    Args:
        leading_trivia: Optional leading trivia preceding this node.
        trailing_trivia: Optional trailing trivia following this node.
        **kwargs: Subclass field values forwarded to CSTNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)

  @classmethod
  def parse(cls, code: str) -> "CppNode":
    """Parse a string of C++ code into a CppNode tree.

    Args:
        code: The C++ source code to parse.

    Returns:
        CppNode: The root of the parsed AST.
    """
    from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser

    parser = CppParser(code)
    return parser.parse()

  def to_text(self) -> str:
    """Render the node to a C++ code string.

    Returns:
        str: The rendered C++ source string.

    Raises:
        NotImplementedError: For the base class.
    """
    raise NotImplementedError("Subclasses must implement to_text()")


class TypeIdentifier(CppNode):
  """Represent a C++ type identifier (e.g., 'int', 'torch::Tensor').

  Attributes:
      name: The name (`str`).
  """

  name: str

  def __init__(self, name: str) -> None:
    """Initialize TypeIdentifier.

    Args:
        name: The C++ type name.
    """
    super().__init__(name=name)

  def to_text(self) -> str:
    """Return the type name.

    Returns:
        str: The type name.
    """
    return self.name


class Expression(CppNode):
  """Base class for all C++ expressions."""

  def __init__(
    self,
    leading_trivia: TriviaInput = None,
    trailing_trivia: TriviaInput = None,
    **kwargs: CstFieldValue,
  ) -> None:
    """Initialize Expression.

    Args:
        leading_trivia: Optional leading trivia preceding this expression.
        trailing_trivia: Optional trailing trivia following this expression.
        **kwargs: Subclass field values forwarded to CppNode.
    """
    super().__init__(leading_trivia=leading_trivia, trailing_trivia=trailing_trivia, **kwargs)


class Identifier(Expression):
  """Represent a simple variable or identifier.

  Attributes:
      name: The name (`str`).
  """

  name: str

  def __init__(self, name: str) -> None:
    """Initialize Identifier.

    Args:
        name: Variable or identifier name.
    """
    super().__init__(name=name)

  def to_text(self) -> str:
    """Return the identifier name as text.

    Returns:
        str: The identifier name.
    """
    return self.name


class BinaryExpression(Expression):
  """Represent a binary expression like a + b.

  Attributes:
      left: The left (`Expression`).
      operator: The operator (`str`).
      right: The right (`Expression`).
  """

  left: Expression
  operator: str
  right: Expression

  def __init__(self, left: Expression, operator: str, right: Expression) -> None:
    """Initialize BinaryExpression.

    Args:
        left: Left-hand operand expression.
        operator: Binary operator string.
        right: Right-hand operand expression.
    """
    super().__init__(left=left, operator=operator, right=right)

  def to_text(self) -> str:
    """Return the binary expression as text.

    Returns:
        str: The binary expression.
    """
    return f"{self.left.to_text()} {self.operator} {self.right.to_text()}"


class MethodCall(Expression):
  """Represent a method call or function invocation.

  Attributes:
      name: The name (`str`).
      arguments: The arguments (`list[Expression]`).
  """

  name: str
  arguments: list[Expression] = pydantic.Field(default_factory=list)

  def __init__(self, name: str, arguments: typing.Sequence[Expression] | None = None) -> None:
    """Initialize MethodCall.

    Args:
        name: Method or function name.
        arguments: List of argument expressions.
    """
    super().__init__(name=name, arguments=list(arguments) if arguments is not None else [])

  def to_text(self) -> str:
    """Return the method call as text.

    Returns:
        str: The method call.
    """
    args_str = ", ".join(a.to_text() for a in self.arguments)
    return f"{self.name}({args_str})"


class ReturnStatement(CppNode):
  """Represent a return statement.

  Attributes:
      value: The value (`Expression | None`).
  """

  value: Expression | None = None

  def __init__(self, value: Expression | None = None) -> None:
    """Initialize ReturnStatement.

    Args:
        value: Optional return value expression.
    """
    super().__init__(value=value)

  def to_text(self) -> str:
    """Return the return statement as text.

    Returns:
        str: The return statement.
    """
    if self.value:
      return f"return {self.value.to_text()};"
    return "return;"


class VariableDeclaration(CppNode):
  """Represent a variable declaration statement.

  Attributes:
      type_id: The type id (`TypeIdentifier`).
      name: The name (`str`).
      initializer: The initializer (`str | Expression | None`).
  """

  type_id: TypeIdentifier
  name: str
  initializer: str | Expression | None = None

  def __init__(
    self,
    type_id: TypeIdentifier,
    name: str,
    initializer: str | Expression | None = None,
  ) -> None:
    """Initialize VariableDeclaration.

    Args:
        type_id: Variable type identifier.
        name: Variable name.
        initializer: Optional initializer expression or string.
    """
    super().__init__(type_id=type_id, name=name, initializer=initializer)

  def to_text(self) -> str:
    """Render the declaration.

    Returns:
        str: The declaration.
    """
    base = f"{self.type_id.to_text()} {self.name}"
    if self.initializer:
      init_text = self.initializer.to_text() if isinstance(self.initializer, Expression) else self.initializer
      return f"{base} = {init_text};"
    return f"{base};"


class FunctionArgument(CppNode):
  """Represent an argument in a function signature.

  Attributes:
      type_id: The type id (`TypeIdentifier`).
      name: The name (`str`).
  """

  type_id: TypeIdentifier
  name: str

  def __init__(self, type_id: TypeIdentifier, name: str) -> None:
    """Initialize FunctionArgument.

    Args:
        type_id: Argument type identifier.
        name: Argument name.
    """
    super().__init__(type_id=type_id, name=name)

  def to_text(self) -> str:
    """Render the argument.

    Returns:
        str: The argument.
    """
    return f"{self.type_id.to_text()} {self.name}"


class FunctionDefinition(CppNode):
  """Represent a C++ function definition.

  Attributes:
      return_type: The return type (`TypeIdentifier`).
      name: The name (`str`).
      arguments: The arguments (`list[FunctionArgument]`).
      body: The body (`list[CppNode]`).
  """

  return_type: TypeIdentifier
  name: str
  arguments: list[FunctionArgument]
  body: list[CppNode] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    return_type: TypeIdentifier,
    name: str,
    arguments: typing.Sequence[FunctionArgument],
    body: typing.Sequence[CppNode] | None = None,
  ) -> None:
    """Initialize FunctionDefinition.

    Args:
        return_type: Function return type identifier.
        name: Function name.
        arguments: List of function arguments.
        body: List of body statements.
    """
    super().__init__(
      return_type=return_type,
      name=name,
      arguments=list(arguments),
      body=list(body) if body is not None else [],
    )

  def to_text(self) -> str:
    """Render the function.

    Returns:
        str: The function.
    """
    args_str = ", ".join(arg.to_text() for arg in self.arguments)
    lines = [f"{self.return_type.to_text()} {self.name}({args_str}) {{"]
    lines.extend(f"    {stmt.to_text()}" for stmt in self.body)
    lines.append("}")
    return "\n".join(lines)


class RawStatement(CppNode):
  """Raw C++ statement to ease transition before full CST modeling.

  Attributes:
      code: The code (`str`).
  """

  code: str

  def __init__(self, code: str) -> None:
    """Initialize RawStatement.

    Args:
        code: Raw C++ statement string.
    """
    super().__init__(code=code)

  def to_text(self) -> str:
    """Render the raw code.

    Returns:
        str: The raw code.
    """
    return self.code


class MacroDefinition(CppNode):
  """Represent a preprocessor macro definition.

  Attributes:
      name: The name (`str`).
      value: The value (`str`).
  """

  name: str
  value: str

  def __init__(self, name: str, value: str) -> None:
    """Initialize MacroDefinition.

    Args:
        name: Macro name.
        value: Macro value string.
    """
    super().__init__(name=name, value=value)

  def to_text(self) -> str:
    """Render the macro definition.

    Returns:
        str: The macro definition.
    """
    return f"#define {self.name} {self.value}"


class BlockStatement(CppNode):
  """Represent a block of C++ code wrapped in braces.

  Attributes:
      statements: The statements (`list[CppNode]`).
  """

  statements: list[CppNode]

  def __init__(self, statements: typing.Sequence[CppNode]) -> None:
    """Initialize BlockStatement.

    Args:
        statements: List of statements inside the block.
    """
    super().__init__(statements=list(statements))

  def to_text(self) -> str:
    """Render the block.

    Returns:
        str: The block.
    """
    lines = ["{"]
    lines.extend(f"    {s.to_text()}" for s in self.statements)
    lines.append("}")
    return "\n".join(lines)


class PyBindDef(CppNode):
  """Represent a pybind11 module method definition.

  Attributes:
      name: The name (`str`).
      function_ref: The function ref (`str`).
      docstring: The docstring (`str`).
  """

  name: str
  function_ref: str
  docstring: str

  def __init__(self, name: str, function_ref: str, docstring: str) -> None:
    """Initialize PyBindDef.

    Args:
        name: Python method name.
        function_ref: C++ function reference name.
        docstring: Method docstring.
    """
    super().__init__(name=name, function_ref=function_ref, docstring=docstring)

  def to_text(self) -> str:
    """Render the m.def() call.

    Returns:
        str: The m.def() call.
    """
    return f'm.def("{self.name}", &{self.function_ref}, "{self.docstring}");'


class PyBindModule(CppNode):
  """Represent a PYBIND11_MODULE block.

  Attributes:
      name: The name (`str`).
      module_var: The module var (`str`).
      defs: The defs (`list[PyBindDef]`).
  """

  name: str
  module_var: str
  defs: list[PyBindDef]

  def __init__(self, name: str, module_var: str, defs: typing.Sequence[PyBindDef]) -> None:
    """Initialize PyBindModule.

    Args:
        name: PyBind module name.
        module_var: Module variable identifier.
        defs: List of PyBindDef bindings.
    """
    super().__init__(name=name, module_var=module_var, defs=list(defs))

  def to_text(self) -> str:
    """Render the PYBIND11_MODULE block.

    Returns:
        str: The module block.
    """
    lines = [f"PYBIND11_MODULE({self.name}, {self.module_var}) {{"]
    lines.extend(f"    {d.to_text()}" for d in self.defs)
    lines.append("}")
    return "\n".join(lines)


class IncludeDirective(CppNode):
  """Represent an #include directive.

  Attributes:
      path: The path (`str`).
      system: The system (`bool`).
  """

  path: str
  system: bool = False

  def __init__(self, path: str, system: bool = False) -> None:
    """Initialize IncludeDirective.

    Args:
        path: Header path.
        system: Whether to use angle brackets for system includes.

    Raises:
        ValueError: If path is invalid.
    """
    if not path or not str(path).strip():
      raise ValueError("Include path must be a non-empty string.")
    if "<" in path or ">" in path or '"' in path:
      raise ValueError('Include path must not contain delimiters like <, >, or ".')
    super().__init__(path=path, system=system)

  def to_text(self) -> str:
    """Render the include.

    Returns:
        str: The include directive.
    """
    if self.system:
      return f"#include <{self.path}>"
    return f'#include "{self.path}"'


class CppModule(CppNode):
  """Represent a complete C++ source file.

  Attributes:
      includes: The includes (`list[IncludeDirective]`).
      body: The body (`list[CppNode]`).
  """

  includes: list[IncludeDirective] = pydantic.Field(default_factory=list)
  body: list[CppNode] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    includes: typing.Sequence[IncludeDirective] | None = None,
    body: typing.Sequence[CppNode] | None = None,
  ) -> None:
    """Initialize CppModule.

    Args:
        includes: List of include directives.
        body: List of top-level C++ nodes.
    """
    super().__init__(
      includes=list(includes) if includes is not None else [],
      body=list(body) if body is not None else [],
    )

  def to_text(self) -> str:
    """Render the entire module.

    Returns:
        str: The module source code.
    """
    lines = [inc.to_text() for inc in self.includes]
    if lines:
      lines.append("")
    for node in self.body:
      lines.append(node.to_text())
      lines.append("")
    return "\n".join(lines).strip() + "\n"
