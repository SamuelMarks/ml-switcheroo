r"""MIDL Semantic Nodes.

This module defines the data structures representing the primitives of the
LaTeX DSL. These nodes act as an intermediate representation between
raw LaTeX macros and the compiler's logical graph.

Classes match the DSL macros:
    - ``ModelContainer`` -> ``\\begin{DefModel}``
    - ``MemoryNode``     -> ``\\Attribute``
    - ``InputNode``      -> ``\\Input``
    - ``ComputeNode``    -> ``\\Op``
    - ``StateOpNode``    -> ``\\StateOp``
    - ``ReturnNode``     -> ``\\Return``
"""

from abc import ABC
import typing

import pydantic

from ml_switcheroo.core.markup.cst import MarkupNode
from ml_switcheroo.utils.formatting import StructuredFormatter, escape_latex


class LatexNode(MarkupNode, ABC):
  """Abstract base class for all MIDL nodes.

  Enforces a ``emit()`` method for serialization support.
  """

  def emit(self, indent_level: int = 0) -> str:
    """Serialize the node object back into its LaTeX macro representation with indentation.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: Valid LaTeX code string.
    """
    return ""

  def to_latex(self) -> str:
    """Provide method to render to LaTeX string.

    Returns:
        str: Valid LaTeX code string representing the node.
    """
    return self.emit(0)


class TextNode(LatexNode):
  """Raw text node for comments and literals."""

  content: str
  """Execute raw text content of the node."""

  def __init__(self, content: str) -> None:
    """Initialize TextNode.

    Args:
        content: Raw text content.
    """
    pydantic.BaseModel.__init__(self, content=content)

  def emit(self, indent_level: int = 0) -> str:
    """Emit raw text.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The raw text representation.
    """
    fmt = StructuredFormatter()
    fmt.add_line(self.content, indent_level)
    return fmt.build()


class MacroNode(LatexNode):
  r"""Represent a generic LaTeX macro (e.g. \Macro{arg1}{arg2})."""

  name: str
  """Execute name of the LaTeX macro."""
  args: typing.List[str] = pydantic.Field(default_factory=list)
  """Argument passed inside curly braces ``{}``."""
  options: typing.List[str] = pydantic.Field(default_factory=list)
  """Optional arguments passed inside square brackets ``[]``."""

  def __init__(
    self,
    name: str,
    args: typing.Optional[typing.List[str]] = None,
    options: typing.Optional[typing.List[str]] = None,
  ) -> None:
    """Initialize MacroNode.

    Args:
        name: Name of the LaTeX macro.
        args: Arguments inside curly braces.
        options: Optional arguments inside square brackets.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      args=args if args is not None else [],
      options=options if options is not None else [],
    )

  def emit(self, indent_level: int = 0) -> str:
    """Render the macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: Renders macro representation as valid LaTeX.
    """
    fmt = StructuredFormatter()
    opts = f"[{', '.join(self.options)}]" if self.options else ""
    args_str = "".join(f"{{{a}}}" for a in self.args)
    fmt.add_line(f"\\{self.name}{opts}{args_str}", indent_level)
    return fmt.build()


class EnvironmentNode(LatexNode):
  r"""Represent a LaTeX environment (e.g. \begin{env}...\end{env})."""

  name: str
  """Execute name of the LaTeX environment."""
  args: typing.List[str] = pydantic.Field(default_factory=list)
  """Argument for the environment block inside curly braces ``{}``."""
  children: typing.List[LatexNode] = pydantic.Field(default_factory=list)
  """Child nodes nested within the environment."""

  def __init__(
    self,
    name: str,
    args: typing.Optional[typing.List[str]] = None,
    children: typing.Optional[typing.List[LatexNode]] = None,
  ) -> None:
    """Initialize EnvironmentNode.

    Args:
        name: Name of the LaTeX environment.
        args: Arguments inside curly braces.
        children: Child nodes nested within the environment.
    """
    pydantic.BaseModel.__init__(
      self,
      name=name,
      args=args if args is not None else [],
      children=children if children is not None else [],
    )

  def emit(self, indent_level: int = 0) -> str:
    """Render the environment.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The environment representation as valid LaTeX.
    """
    fmt = StructuredFormatter()
    args_str = "".join(f"{{{a}}}" for a in self.args)
    fmt.add_line(f"\\begin{{{self.name}}}{args_str}", indent_level)
    for child in self.children:
      fmt.add_line(child.emit(indent_level + 1), 0)
    fmt.add_line(f"\\end{{{self.name}}}", indent_level)
    return fmt.build()


class MemoryNode(LatexNode):
  r"""Represent stateful memory allocation (e.g., Weights/Layers).

  Maps to the ``\\Attribute`` macro.

  Example::

      \\Attribute{conv}{Conv2d}{in=1, out=32, k=3}
  """

  node_id: str
  """Execute unique identifier for the attribute."""

  op_type: str
  """Execute operation type (e.g., 'Conv2d')."""

  config: typing.Dict[str, str] = pydantic.Field(default_factory=dict)
  """Configuration parameters for the layer."""

  def __init__(
    self,
    node_id: str,
    op_type: str,
    config: typing.Optional[typing.Dict[str, str]] = None,
  ) -> None:
    """Initialize MemoryNode.

    Args:
        node_id: Unique identifier for the attribute.
        op_type: Operation type.
        config: Configuration parameters for the layer.
    """
    pydantic.BaseModel.__init__(
      self,
      node_id=node_id,
      op_type=op_type,
      config=config if config is not None else {},
    )

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Attribute`` macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The attribute macro representation as valid LaTeX.
    """
    config_str = ", ".join(f"{escape_latex(k)}={escape_latex(v)}" for k, v in self.config.items())
    safe_id = escape_latex(self.node_id)
    safe_op = escape_latex(self.op_type)
    return MacroNode("Attribute", [safe_id, safe_op, config_str]).emit(indent_level)


class InputNode(LatexNode):
  r"""Represent the model input definition.

  Maps to the ``\\Input`` macro.

  Example::

      \\Input{x}{[B, 1, 28, 28]}
  """

  name: str
  """Name of the input variable."""

  shape: str
  """Shape descriptor string."""

  def __init__(self, name: str, shape: str) -> None:
    """Initialize InputNode.

    Args:
        name: Name of the input variable.
        shape: Shape descriptor string.
    """
    pydantic.BaseModel.__init__(self, name=name, shape=shape)

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Input`` macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The input macro representation as valid LaTeX.
    """
    safe_name = escape_latex(self.name)
    safe_shape = escape_latex(self.shape)
    return MacroNode("Input", [safe_name, safe_shape]).emit(indent_level)


class ComputeNode(LatexNode):
  r"""Represent a stateless operation call.

  Maps to the ``\\Op`` macro.

  Example::

      \\Op{s2}{Flatten}{s1, start=1}{[B, 21632]}
  """

  node_id: str
  """Execute unique identifier to assign the result to."""

  op_type: str
  """Execute operation type (e.g., 'Flatten')."""

  args: typing.List[str]
  """List of arguments passed to the operation."""

  shape: str
  """Resulting shape descriptor."""

  def __init__(self, node_id: str, op_type: str, args: typing.List[str], shape: str) -> None:
    """Initialize ComputeNode.

    Args:
        node_id: Unique identifier to assign the result to.
        op_type: Operation type.
        args: List of arguments passed to the operation.
        shape: Resulting shape descriptor.
    """
    pydantic.BaseModel.__init__(self, node_id=node_id, op_type=op_type, args=args, shape=shape)

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Op`` macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The compute operation macro representation as valid LaTeX.
    """
    args_str = ", ".join(escape_latex(a) for a in self.args)
    safe_id = escape_latex(self.node_id)
    safe_op = escape_latex(self.op_type)
    safe_shape = escape_latex(self.shape)
    return MacroNode("Op", [safe_id, safe_op, args_str, safe_shape]).emit(indent_level)


class StateOpNode(LatexNode):
  r"""Represent a call to a stateful layer defined in Memory.

  Maps to the ``\\StateOp`` macro.

  Example::

      \\StateOp{s1}{conv}{x}{[B, 32, 26, 26]}
  """

  node_id: str
  """Execute unique identifier to assign the result to."""

  attribute_id: str
  """Execute ID of the attribute being called."""

  args: typing.List[str]
  """List of arguments passed to the call."""

  shape: str
  """Resulting shape descriptor."""

  def __init__(self, node_id: str, attribute_id: str, args: typing.List[str], shape: str) -> None:
    """Initialize StateOpNode.

    Args:
        node_id: Unique identifier to assign the result to.
        attribute_id: ID of the attribute being called.
        args: List of arguments passed to the call.
        shape: Resulting shape descriptor.
    """
    pydantic.BaseModel.__init__(self, node_id=node_id, attribute_id=attribute_id, args=args, shape=shape)

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\StateOp`` macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The stateful operation macro representation as valid LaTeX.
    """
    args_str = ", ".join(escape_latex(a) for a in self.args)
    safe_id = escape_latex(self.node_id)
    safe_attr = escape_latex(self.attribute_id)
    safe_shape = escape_latex(self.shape)
    return MacroNode("StateOp", [safe_id, safe_attr, args_str, safe_shape]).emit(indent_level)


class ReturnNode(LatexNode):
  r"""Represent the output return statement.

  Maps to the ``\\Return`` macro.

  Example::

      \\Return{s3}
  """

  target_id: str
  """Execute variable ID to return."""

  def __init__(self, target_id: str) -> None:
    """Initialize ReturnNode.

    Args:
        target_id: Variable ID to return.
    """
    pydantic.BaseModel.__init__(self, target_id=target_id)

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Return`` macro.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The return macro representation as valid LaTeX.
    """
    safe_tgt = escape_latex(self.target_id)
    return MacroNode("Return", [safe_tgt]).emit(indent_level)


class DocumentNode(LatexNode):
  """Root container representing the full LaTeX document."""

  children: typing.List[LatexNode] = pydantic.Field(default_factory=list)
  """List of child nodes comprising the document."""

  def __init__(self, children: typing.Optional[typing.List[LatexNode]] = None) -> None:
    """Initialize DocumentNode.

    Args:
        children: List of child nodes comprising the document.
    """
    pydantic.BaseModel.__init__(self, children=children if children is not None else [])

  def emit(self, indent_level: int = 0) -> str:
    """Emit the sequence of document nodes.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The concatenated LaTeX code for all child nodes.
    """
    return "\n".join(child.emit(indent_level) for child in self.children)


class ModelContainer(LatexNode):
  """Root container representing the Model definition block.

  Maps to the ``DefModel`` environment.
  """

  name: str
  """Execute model class name."""

  children: typing.List[LatexNode] = pydantic.Field(default_factory=list)
  """List of body statements (Memory, Input, Ops, Return)."""

  def __init__(self, name: str, children: typing.Optional[typing.List[LatexNode]] = None) -> None:
    """Initialize ModelContainer.

    Args:
        name: Model class name.
        children: List of body statements.
    """
    pydantic.BaseModel.__init__(self, name=name, children=children if children is not None else [])

  def emit(self, indent_level: int = 0) -> str:
    r"""Render the full ``\\begin{DefModel}...\\end{DefModel}`` block.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: The model container block representation as valid LaTeX.
    """
    safe_name = escape_latex(self.name)
    return EnvironmentNode("DefModel", [safe_name], self.children).emit(indent_level)
