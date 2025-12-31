"""
TikZ Concrete Syntax Tree (CST) Data Structures.

This package provides a pure Python representation of TikZ/PGF code designed to
represent neural network graphs with high fidelity. It supports:
- Nodes with HTML-like label tables for layer metadata.
- Edges representing data flow.
- Trivia preservation (comments/whitespace) for round-trip stability.
"""

from ml_switcheroo.core.tikz.nodes import (
  TikzBaseNode,
  TriviaNode,
  TikzComment,
  TikzOption,
  TikzTable,
  TikzNode,
  TikzEdge,
  TikzGraph,
)

__all__ = [
  "TikzBaseNode",
  "TriviaNode",
  "TikzComment",
  "TikzOption",
  "TikzTable",
  "TikzNode",
  "TikzEdge",
  "TikzGraph",
]
