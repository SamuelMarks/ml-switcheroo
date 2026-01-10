"""
RDNA AST Nodes (Shim).

This module aliases the unified AST nodes from the compiler frontend package
to maintain backward compatibility with previous core tests.
"""

from ml_switcheroo.compiler.frontends.rdna.nodes import (
  RdnaNode,
  Operand,
  LabelRef,
  SGPR,
  VGPR,
  c_SGPR,
  c_VGPR,
  Immediate,
  Modifier,
  Memory,
  Instruction,
  Label,
  Directive,
  Comment,
)

__all__ = [
  "RdnaNode",
  "Operand",
  "LabelRef",
  "SGPR",
  "VGPR",
  "c_SGPR",
  "c_VGPR",
  "Immediate",
  "Modifier",
  "Memory",
  "Instruction",
  "Label",
  "Directive",
  "Comment",
]
