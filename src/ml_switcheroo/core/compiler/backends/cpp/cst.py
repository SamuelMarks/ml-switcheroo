"""C++ Concrete Syntax Tree (CST) Builder API.

Provides a structural way to generate C++ code, avoiding brittle string
concatenation and macros. Inspired by `libcst` for Python.
"""

from dataclasses import dataclass, field
from typing import List, Optional


class CppNode:
  """Base class for all C++ CST nodes."""

  def to_text(self) -> str:
    """Renders the node to a C++ code string.

    Returns:
        str: C++ code.
    """
    raise NotImplementedError("Subclasses must implement to_text()")


@dataclass
class TypeIdentifier(CppNode):
  """Represents a C++ type identifier (e.g., 'int', 'torch::Tensor')."""

  name: str

  def to_text(self) -> str:
    """Returns the type name."""
    return self.name


@dataclass
class VariableDeclaration(CppNode):
  """Represents a variable declaration."""

  type_id: TypeIdentifier
  name: str
  initializer: Optional[str] = None

  def to_text(self) -> str:
    """Renders the declaration."""
    base = f"{self.type_id.to_text()} {self.name}"
    if self.initializer:
      return f"{base} = {self.initializer};"
    return f"{base};"


@dataclass
class FunctionArgument(CppNode):
  """Represents an argument in a function signature."""

  type_id: TypeIdentifier
  name: str

  def to_text(self) -> str:
    """Renders the argument."""
    return f"{self.type_id.to_text()} {self.name}"


@dataclass
class FunctionDefinition(CppNode):
  """Represents a C++ function definition."""

  return_type: TypeIdentifier
  name: str
  arguments: List[FunctionArgument]
  body: List[CppNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the function."""
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
    """Renders the raw code."""
    return self.code


@dataclass
class IncludeDirective(CppNode):
  """Represents an #include directive."""

  path: str
  system: bool = False

  def to_text(self) -> str:
    """Renders the include."""
    if self.system:
      return f"#include <{self.path}>"
    return f'#include "{self.path}"'


@dataclass
class CppModule(CppNode):
  """Represents a complete C++ source file."""

  includes: List[IncludeDirective] = field(default_factory=list)
  body: List[CppNode] = field(default_factory=list)

  def to_text(self) -> str:
    """Renders the entire module."""
    lines = [inc.to_text() for inc in self.includes]
    if lines:
      lines.append("")
    for node in self.body:
      lines.append(node.to_text())
      lines.append("")
    return "\n".join(lines).strip() + "\n"
