# src/ml_switcheroo/core/latex/nodes.py

"""
MIDL Semantic Nodes.

This module defines the data structures representing the primitives of the
LaTeX DSL. These nodes act as an intermediate representation between
raw LaTeX macros and the compiler's logical graph.

Classes match the DSL macros:
    - ModelContainer -> \\begin{DefModel}
    - MemoryNode     -> \\Attribute
    - InputNode      -> \\Input
    - ComputeNode    -> \\Op
    - StateOpNode    -> \\StateOp
    - ReturnNode     -> \\Return
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional


@dataclass
class LatexNode(ABC):
  """
  Abstract base class for all MIDL nodes.

  Enforces a `to_latex()` method for serialization support.
  """

  @abstractmethod
  def to_latex(self) -> str:
    """
    Serializes the node object back into its LaTeX macro representation.

    Returns:
        str: Valid LaTeX code string.
    """
    pass


@dataclass
class MemoryNode(LatexNode):
  """
  Represents stateful memory allocation (e.g., Weights/Layers).
  Maps to the `\\Attribute` macro.

  Example:
      \\Attribute{conv}{Conv2d}{in=1, out=32, k=3}
  """

  node_id: str
  op_type: str
  config: Dict[str, str] = field(default_factory=dict)

  def to_latex(self) -> str:
    """Render to \\Attribute macro."""
    # Convert config dict to string "k=v, k2=v2"
    config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
    return f"\\Attribute{{{self.node_id}}}{{{self.op_type}}}{{{config_str}}}"


@dataclass
class InputNode(LatexNode):
  """
  Represents the model input definition.
  Maps to the `\\Input` macro.

  Example:
      \\Input{x}{[B, 1, 28, 28]}
  """

  name: str
  shape: str

  def to_latex(self) -> str:
    """Render to \\Input macro."""
    return f"\\Input{{{self.name}}}{{{self.shape}}}"


@dataclass
class ComputeNode(LatexNode):
  """
  Represents a stateless operation call.
  Maps to the `\\Op` macro.

  Example:
      \\Op{s2}{Flatten}{s1, start=1}{[B, 21632]}
  """

  node_id: str
  op_type: str
  args: List[str]
  shape: str

  def to_latex(self) -> str:
    """Render to \\Op macro."""
    args_str = ", ".join(self.args)
    return f"\\Op{{{self.node_id}}}{{{self.op_type}}}{{{args_str}}}{{{self.shape}}}"


@dataclass
class StateOpNode(LatexNode):
  """
  Represents a call to a stateful layer defined in Memory.
  Maps to the `\\StateOp` macro.

  Example:
      \\StateOp{s1}{conv}{x}{[B, 32, 26, 26]}
  """

  node_id: str
  attribute_id: str
  args: List[str]
  shape: str

  def to_latex(self) -> str:
    """Render to \\StateOp macro."""
    args_str = ", ".join(self.args)
    return f"\\StateOp{{{self.node_id}}}{{{self.attribute_id}}}{{{args_str}}}{{{self.shape}}}"


@dataclass
class ReturnNode(LatexNode):
  """
  Represents the output return statement.
  Maps to the `\\Return` macro.

  Example:
      \\Return{s3}
  """

  target_id: str

  def to_latex(self) -> str:
    """Render to \\Return macro."""
    return f"\\Return{{{self.target_id}}}"


@dataclass
class ModelContainer(LatexNode):
  """
  Root container representing the Model definition block.
  Maps to the `DefModel` environment.

  Attributes:
      name: The model class name.
      children: List of body statements (Memory, Input, Ops, Return).
  """

  name: str
  children: List[LatexNode] = field(default_factory=list)

  def to_latex(self) -> str:
    """Render the full \\begin{DefModel}...\\end{DefModel} block."""
    lines = [f"\\begin{{DefModel}}{{{self.name}}}"]

    # Indent children
    for child in self.children:
      lines.append(f"    {child.to_latex()}")

    lines.append(r"\end{DefModel}")
    return "\n".join(lines)
