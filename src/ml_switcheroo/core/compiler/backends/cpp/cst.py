"""C++ Concrete Syntax Tree (CST) Builder API.

Provides a structural way to generate C++ code, avoiding brittle string
concatenation and macros. Inspired by `libcst` for Python.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


class CppNode:
  """Base class for all C++ CST nodes."""

  @classmethod
  def parse(cls, code: str) -> "CppNode":
    """Parses a string of C++ code into a CppNode tree.

    Args:
        code: The C++ source code to parse.

    Returns:
        CppNode: The root of the parsed AST.
    """
    from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser

    parser = CppParser(code)
    return parser.parse()

  def to_text(self) -> str:
    """Renders the node to a C++ code string.

    Raises:
        NotImplementedError: For the base class.
    """
    raise NotImplementedError("Subclasses must implement to_text()")


@dataclass
class TypeIdentifier(CppNode):
  """Represents a C++ type identifier (e.g., 'int', 'torch::Tensor')."""

  name: str

  def to_text(self) -> str:
    """Returns the type name.

    Returns:
        str: The type name.
    """
    return self.name


@dataclass
class Expression(CppNode):
  """Base class for all C++ expressions."""

  pass


@dataclass
class Identifier(Expression):
  """Represents a simple variable or identifier."""

  name: str

  def to_text(self) -> str:
    """Returns the identifier name as text.

    Returns:
        str: The identifier name.
    """
    return self.name


@dataclass
class BinaryExpression(Expression):
  """Represents a binary expression like a + b."""

  left: Expression
  operator: str
  right: Expression

  def to_text(self) -> str:
    """Returns the binary expression as text.

    Returns:
        str: The binary expression.
    """
    return f"{self.left.to_text()} {self.operator} {self.right.to_text()}"


@dataclass
class MethodCall(Expression):
  """Represents a method call or function invocation."""

  name: str
  arguments: List[Expression] = field(default_factory=list)

  def to_text(self) -> str:
    """Returns the method call as text.

    Returns:
        str: The method call.
    """
    args_str = ", ".join(a.to_text() for a in self.arguments)
    return f"{self.name}({args_str})"


@dataclass
class ReturnStatement(CppNode):
  """Represents a return statement."""

  value: Optional[Expression] = None

  def to_text(self) -> str:
    """Returns the return statement as text.

    Returns:
        str: The return statement.
    """
    if self.value:
      return f"return {self.value.to_text()};"
    return "return;"


@dataclass
class VariableDeclaration(CppNode):
  """Represents a variable declaration."""

  type_id: TypeIdentifier
  name: str
  initializer: Optional[Union[str, Expression]] = None

  def to_text(self) -> str:
    """Renders the declaration.

    Returns:
        str: The declaration.
    """
    base = f"{self.type_id.to_text()} {self.name}"
    if self.initializer:
      init_text = self.initializer.to_text() if isinstance(self.initializer, Expression) else self.initializer
      return f"{base} = {init_text};"
    return f"{base};"


@dataclass
class FunctionArgument(CppNode):
  """Represents an argument in a function signature."""

  type_id: TypeIdentifier
  name: str

  def to_text(self) -> str:
    """Renders the argument.

    Returns:
        str: The argument.
    """
    return f"{self.type_id.to_text()} {self.name}"


@dataclass
class FunctionDefinition(CppNode):
  """Represents a C++ function definition."""

  return_type: TypeIdentifier
  name: str
  arguments: List[FunctionArgument]
  body: List[CppNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the function.

    Returns:
        str: The function.
    """
    args_str = ", ".join(arg.to_text() for arg in self.arguments)
    lines = [f"{self.return_type.to_text()} {self.name}({args_str}) {{"]
    for stmt in self.body:
      lines.append(f"    {stmt.to_text()}")
    lines.append("}")
    return "\n".join(lines)


@dataclass
class RawStatement(CppNode):
  """Raw C++ statement to ease transition before full CST modeling."""

  code: str

  def to_text(self) -> str:
    """Renders the raw code.

    Returns:
        str: The raw code.
    """
    return self.code


@dataclass
class MacroDefinition(CppNode):
  """Represents a preprocessor macro definition."""

  name: str
  value: str

  def to_text(self) -> str:
    """Renders the macro definition.

    Returns:
        str: The macro definition.
    """
    return f"#define {self.name} {self.value}"


@dataclass
class BlockStatement(CppNode):
  """Represents a block of C++ code wrapped in braces."""

  statements: List[CppNode]

  def to_text(self) -> str:
    """Renders the block.

    Returns:
        str: The block.
    """
    lines = ["{"]
    for s in self.statements:
      lines.append(f"    {s.to_text()}")
    lines.append("}")
    return "\n".join(lines)


@dataclass
class PyBindDef(CppNode):
  """Represents a pybind11 module method definition."""

  name: str
  function_ref: str
  docstring: str

  def to_text(self) -> str:
    """Renders the m.def() call.

    Returns:
        str: The m.def() call.
    """
    return f'm.def("{self.name}", &{self.function_ref}, "{self.docstring}");'


@dataclass
class PyBindModule(CppNode):
  """Represents a PYBIND11_MODULE block."""

  name: str
  module_var: str
  defs: List[PyBindDef]

  def to_text(self) -> str:
    """Renders the PYBIND11_MODULE block.

    Returns:
        str: The module block.
    """
    lines = [f"PYBIND11_MODULE({self.name}, {self.module_var}) {{"]
    for d in self.defs:
      lines.append(f"    {d.to_text()}")
    lines.append("}")
    return "\n".join(lines)


@dataclass
class IncludeDirective(CppNode):
  """Represents an #include directive."""

  path: str
  system: bool = False

  def __post_init__(self) -> None:
    """Enforces typing rules.

    Raises:
        ValueError: If path is invalid.
    """
    if not isinstance(self.path, str) or not self.path.strip():
      raise ValueError("Include path must be a non-empty string.")
    if "<" in self.path or ">" in self.path or '"' in self.path:
      raise ValueError('Include path must not contain delimiters like <, >, or ".')

  def to_text(self) -> str:
    """Renders the include.

    Returns:
        str: The include directive.
    """
    if self.system:
      return f"#include <{self.path}>"
    return f'#include "{self.path}"'


@dataclass
class CppModule(CppNode):
  """Represents a complete C++ source file."""

  includes: List[IncludeDirective] = field(default_factory=list)
  body: List[CppNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the entire module.

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
