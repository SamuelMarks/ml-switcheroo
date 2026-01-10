"""
SASS AST Nodes (Shim).

This module aliases the unified AST nodes from the compiler frontend package
to maintain backward compatibility with previous core tests.
"""

from ml_switcheroo.compiler.frontends.sass.nodes import (
  SassNode,
  Operand,
  Register,
  Predicate,
  Immediate,
  Memory,
  Instruction,
  Label,
  Directive,
  Comment,
  SGPR,
)

__all__ = [
  "SassNode",
  "Operand",
  "Register",
  "Predicate",
  "Immediate",
  "Memory",
  "Instruction",
  "Label",
  "Directive",
  "Comment",
  "SGPR",
]
