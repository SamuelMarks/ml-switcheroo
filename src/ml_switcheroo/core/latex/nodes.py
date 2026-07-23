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
from dataclasses import dataclass, field
from typing import List, Dict
from ml_switcheroo.utils.formatting import StructuredFormatter, escape_latex


@dataclass
class LatexNode(ABC):
  """Abstract base class for all MIDL nodes.

  Enforces a ``emit()`` method for serialization support.
  """

  def emit(self, indent_level: int = 0) -> str:
    """Serializes the node object back into its LaTeX macro representation with indentation.

    Args:
        indent_level: Current indentation depth.

    Returns:
        str: Valid LaTeX code string.
    """
    return ""

  def to_latex(self) -> str:
    """Convenience method to render to LaTeX string."""
    return self.emit(0)


@dataclass
class MemoryNode(LatexNode):
  r"""Represents stateful memory allocation (e.g., Weights/Layers).


  Maps to the ``\\Attribute`` macro.

  Example::

      \\Attribute{conv}{Conv2d}{in=1, out=32, k=3}
  """

  node_id: str
  """The unique identifier for the attribute."""

  op_type: str
  """The operation type (e.g., 'Conv2d')."""

  config: Dict[str, str] = field(default_factory=dict)
  """Configuration parameters for the layer."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Attribute`` macro."""
    fmt = StructuredFormatter()
    # Convert config dict to string "k=v, k2=v2"
    config_str = ", ".join(f"{escape_latex(k)}={escape_latex(v)}" for k, v in self.config.items())
    safe_id = escape_latex(self.node_id)
    safe_op = escape_latex(self.op_type)
    fmt.add_line(f"\\Attribute{{{safe_id}}}{{{safe_op}}}{{{config_str}}}", indent_level)
    return fmt.build()


@dataclass
class InputNode(LatexNode):
  r"""Represents the model input definition.

  Maps to the ``\\Input`` macro.

  Example::

      \\Input{x}{[B, 1, 28, 28]}
  """

  name: str
  """Name of the input variable."""

  shape: str
  """Shape descriptor string."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Input`` macro."""
    fmt = StructuredFormatter()
    safe_name = escape_latex(self.name)
    safe_shape = escape_latex(self.shape)
    fmt.add_line(f"\\Input{{{safe_name}}}{{{safe_shape}}}", indent_level)
    return fmt.build()


@dataclass
class ComputeNode(LatexNode):
  r"""Represents a stateless operation call.

  Maps to the ``\\Op`` macro.

  Example::

      \\Op{s2}{Flatten}{s1, start=1}{[B, 21632]}
  """

  node_id: str
  """The unique identifier to assign the result to."""

  op_type: str
  """The operation type (e.g., 'Flatten')."""

  args: List[str]
  """List of arguments passed to the operation."""

  shape: str
  """Resulting shape descriptor."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Op`` macro."""
    fmt = StructuredFormatter()
    args_str = ", ".join(escape_latex(a) for a in self.args)
    safe_id = escape_latex(self.node_id)
    safe_op = escape_latex(self.op_type)
    safe_shape = escape_latex(self.shape)
    fmt.add_line(f"\\Op{{{safe_id}}}{{{safe_op}}}{{{args_str}}}{{{safe_shape}}}", indent_level)
    return fmt.build()


@dataclass
class StateOpNode(LatexNode):
  r"""Represents a call to a stateful layer defined in Memory.

  Maps to the ``\\StateOp`` macro.

  Example::

      \\StateOp{s1}{conv}{x}{[B, 32, 26, 26]}
  """

  node_id: str
  """The unique identifier to assign the result to."""

  attribute_id: str
  """The ID of the attribute being called."""

  args: List[str]
  """List of arguments passed to the call."""

  shape: str
  """Resulting shape descriptor."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\StateOp`` macro."""
    fmt = StructuredFormatter()
    args_str = ", ".join(escape_latex(a) for a in self.args)
    safe_id = escape_latex(self.node_id)
    safe_attr = escape_latex(self.attribute_id)
    safe_shape = escape_latex(self.shape)
    fmt.add_line(f"\\StateOp{{{safe_id}}}{{{safe_attr}}}{{{args_str}}}{{{safe_shape}}}", indent_level)
    return fmt.build()


@dataclass
class ReturnNode(LatexNode):
  r"""Represents the output return statement.

  Maps to the ``\\Return`` macro.

  Example::

      \\Return{s3}
  """

  target_id: str
  """The variable ID to return."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render to ``\\Return`` macro."""
    fmt = StructuredFormatter()
    safe_tgt = escape_latex(self.target_id)
    fmt.add_line(f"\\Return{{{safe_tgt}}}", indent_level)
    return fmt.build()


@dataclass
class ModelContainer(LatexNode):
  """Root container representing the Model definition block.

  Maps to the ``DefModel`` environment.
  """

  name: str = field()
  """The model class name."""

  children: List[LatexNode] = field(default_factory=list)
  """List of body statements (Memory, Input, Ops, Return)."""

  def emit(self, indent_level: int = 0) -> str:
    r"""Render the full ``\\begin{DefModel}...\\end{DefModel}`` block."""
    fmt = StructuredFormatter()
    safe_name = escape_latex(self.name)
    fmt.add_line(f"\\begin{{DefModel}}{{{safe_name}}}", indent_level)

    for child in self.children:
      fmt.add_line(child.emit(indent_level + 1), 0)

    fmt.add_line(r"\end{DefModel}", indent_level)
    return fmt.build()
