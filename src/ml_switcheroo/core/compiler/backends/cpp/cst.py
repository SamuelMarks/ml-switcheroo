"""C++ Concrete Syntax Tree (CST) Builder API.

Provides a structural way to generate C++ code, avoiding brittle string
concatenation and macros. Inspired by `libcst` for Python.
"""

import typing

import pydantic


class CppNode(pydantic.BaseModel):
  """Base class for all C++ CST nodes."""

  model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
  """Represent a C++ type identifier (e.g., 'int', 'torch::Tensor')."""

  name: str

  def __init__(self, name: str) -> None:
    """Initialize TypeIdentifier.

    Args:
        name: The C++ type name.
    """
    pydantic.BaseModel.__init__(self, name=name)

  def to_text(self) -> str:
    """Return the type name.

    Returns:
        str: The type name.
    """
    return self.name


class Expression(CppNode):
  """Base class for all C++ expressions."""


class Identifier(Expression):
  """Represent a simple variable or identifier."""

  name: str

  def __init__(self, name: str) -> None:
    """Initialize Identifier.

    Args:
        name: Variable or identifier name.
    """
    pydantic.BaseModel.__init__(self, name=name)

  def to_text(self) -> str:
    """Return the identifier name as text.

    Returns:
        str: The identifier name.
    """
    return self.name


class BinaryExpression(Expression):
  """Represent a binary expression like a + b."""

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
    pydantic.BaseModel.__init__(self, left=left, operator=operator, right=right)

  def to_text(self) -> str:
    """Return the binary expression as text.

    Returns:
        str: The binary expression.
    """
    return f"{self.left.to_text()} {self.operator} {self.right.to_text()}"


class MethodCall(Expression):
  """Represent a method call or function invocation."""

  name: str
  arguments: typing.List[Expression] = pydantic.Field(default_factory=list)

  def __init__(self, name: str, arguments: typing.Optional[typing.Sequence[Expression]] = None) -> None:
    """Initialize MethodCall.

    Args:
        name: Method or function name.
        arguments: List of argument expressions.
    """
    pydantic.BaseModel.__init__(self, name=name, arguments=list(arguments) if arguments is not None else [])

  def to_text(self) -> str:
    """Return the method call as text.

    Returns:
        str: The method call.
    """
    args_str = ", ".join(a.to_text() for a in self.arguments)
    return f"{self.name}({args_str})"


class ReturnStatement(CppNode):
  """Represent a return statement."""

  value: typing.Optional[Expression] = None

  def __init__(self, value: typing.Optional[Expression] = None) -> None:
    """Initialize ReturnStatement.

    Args:
        value: Optional return value expression.
    """
    pydantic.BaseModel.__init__(self, value=value)

  def to_text(self) -> str:
    """Return the return statement as text.

    Returns:
        str: The return statement.
    """
    if self.value:
      return f"return {self.value.to_text()};"
    return "return;"


class VariableDeclaration(CppNode):
  """Represent a variable declaration statement."""

  type_id: typing.Annotated[TypeIdentifier, pydantic.SkipValidation]
  name: str
  initializer: typing.Optional[typing.Union[str, Expression]] = None

  def __init__(
    self,
    type_id: TypeIdentifier,
    name: str,
    initializer: typing.Optional[typing.Union[str, Expression]] = None,
  ) -> None:
    """Initialize VariableDeclaration.

    Args:
        type_id: Variable type identifier.
        name: Variable name.
        initializer: Optional initializer expression or string.
    """
    pydantic.BaseModel.__init__(self, type_id=type_id, name=name, initializer=initializer)

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
  """Represent an argument in a function signature."""

  type_id: typing.Annotated[TypeIdentifier, pydantic.SkipValidation]
  name: str

  def __init__(self, type_id: TypeIdentifier, name: str) -> None:
    """Initialize FunctionArgument.

    Args:
        type_id: Argument type identifier.
        name: Argument name.
    """
    pydantic.BaseModel.__init__(self, type_id=type_id, name=name)

  def to_text(self) -> str:
    """Render the argument.

    Returns:
        str: The argument.
    """
    return f"{self.type_id.to_text()} {self.name}"


class FunctionDefinition(CppNode):
  """Represent a C++ function definition."""

  return_type: typing.Annotated[TypeIdentifier, pydantic.SkipValidation]
  name: str
  arguments: typing.Annotated[typing.List[FunctionArgument], pydantic.SkipValidation]
  body: typing.Annotated[typing.List[CppNode], pydantic.SkipValidation] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    return_type: TypeIdentifier,
    name: str,
    arguments: typing.Sequence[FunctionArgument],
    body: typing.Optional[typing.Sequence[CppNode]] = None,
  ) -> None:
    """Initialize FunctionDefinition.

    Args:
        return_type: Function return type identifier.
        name: Function name.
        arguments: List of function arguments.
        body: List of body statements.
    """
    pydantic.BaseModel.__init__(
      self,
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
    for stmt in self.body:
      lines.append(f"    {stmt.to_text()}")
    lines.append("}")
    return "\n".join(lines)


class RawStatement(CppNode):
  """Raw C++ statement to ease transition before full CST modeling."""

  code: str

  def __init__(self, code: str) -> None:
    """Initialize RawStatement.

    Args:
        code: Raw C++ statement string.
    """
    pydantic.BaseModel.__init__(self, code=code)

  def to_text(self) -> str:
    """Render the raw code.

    Returns:
        str: The raw code.
    """
    return self.code


class MacroDefinition(CppNode):
  """Represent a preprocessor macro definition."""

  name: str
  value: str

  def __init__(self, name: str, value: str) -> None:
    """Initialize MacroDefinition.

    Args:
        name: Macro name.
        value: Macro value string.
    """
    pydantic.BaseModel.__init__(self, name=name, value=value)

  def to_text(self) -> str:
    """Render the macro definition.

    Returns:
        str: The macro definition.
    """
    return f"#define {self.name} {self.value}"


class BlockStatement(CppNode):
  """Represent a block of C++ code wrapped in braces."""

  statements: typing.List[CppNode]

  def __init__(self, statements: typing.Sequence[CppNode]) -> None:
    """Initialize BlockStatement.

    Args:
        statements: List of statements inside the block.
    """
    pydantic.BaseModel.__init__(self, statements=list(statements))

  def to_text(self) -> str:
    """Render the block.

    Returns:
        str: The block.
    """
    lines = ["{"]
    for s in self.statements:
      lines.append(f"    {s.to_text()}")
    lines.append("}")
    return "\n".join(lines)


class PyBindDef(CppNode):
  """Represent a pybind11 module method definition."""

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
    pydantic.BaseModel.__init__(self, name=name, function_ref=function_ref, docstring=docstring)

  def to_text(self) -> str:
    """Render the m.def() call.

    Returns:
        str: The m.def() call.
    """
    return f'm.def("{self.name}", &{self.function_ref}, "{self.docstring}");'


class PyBindModule(CppNode):
  """Represent a PYBIND11_MODULE block."""

  name: str
  module_var: str
  defs: typing.List[PyBindDef]

  def __init__(self, name: str, module_var: str, defs: typing.Sequence[PyBindDef]) -> None:
    """Initialize PyBindModule.

    Args:
        name: PyBind module name.
        module_var: Module variable identifier.
        defs: List of PyBindDef bindings.
    """
    pydantic.BaseModel.__init__(self, name=name, module_var=module_var, defs=list(defs))

  def to_text(self) -> str:
    """Render the PYBIND11_MODULE block.

    Returns:
        str: The module block.
    """
    lines = [f"PYBIND11_MODULE({self.name}, {self.module_var}) {{"]
    for d in self.defs:
      lines.append(f"    {d.to_text()}")
    lines.append("}")
    return "\n".join(lines)


class IncludeDirective(CppNode):
  """Represent an #include directive."""

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
    pydantic.BaseModel.__init__(self, path=path, system=system)

  def to_text(self) -> str:
    """Render the include.

    Returns:
        str: The include directive.
    """
    if self.system:
      return f"#include <{self.path}>"
    return f'#include "{self.path}"'


class CppModule(CppNode):
  """Represent a complete C++ source file."""

  includes: typing.List[IncludeDirective] = pydantic.Field(default_factory=list)
  body: typing.Annotated[typing.List[CppNode], pydantic.SkipValidation] = pydantic.Field(default_factory=list)

  def __init__(
    self,
    includes: typing.Optional[typing.Sequence[IncludeDirective]] = None,
    body: typing.Optional[typing.Sequence[CppNode]] = None,
  ) -> None:
    """Initialize CppModule.

    Args:
        includes: List of include directives.
        body: List of top-level C++ nodes.
    """
    pydantic.BaseModel.__init__(
      self,
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
