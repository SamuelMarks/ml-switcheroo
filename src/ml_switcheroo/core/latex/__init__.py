"""
LaTeX DSL Core Package.

This package contains the semantic primitives for the Machine Intelligence Definition Language (MIDL).
It defines the node structures mirroring the LaTeX macros (`\\Attribute`, `\\Op`, etc.) used
to visualize neural network architectures.
"""

from ml_switcheroo.core.latex.nodes import (
  LatexNode,
  ModelContainer,
  MemoryNode,
  InputNode,
  ComputeNode,
  StateOpNode,
  ReturnNode,
)

__all__ = [
  "LatexNode",
  "ModelContainer",
  "MemoryNode",
  "InputNode",
  "ComputeNode",
  "StateOpNode",
  "ReturnNode",
]
