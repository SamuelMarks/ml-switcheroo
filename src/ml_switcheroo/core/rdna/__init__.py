"""
Shim layer for RDNA components.

Re-exports new compiler components for backward compatibility.
"""

from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  LabelRef,
  Memory,
  Modifier,
  Operand,
  RdnaNode,
  SGPR,
  VGPR,
)
from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.compiler.frontends.rdna.tokens import RdnaLexer, Token, TokenType
from ml_switcheroo.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.compiler.frontends.rdna.analysis import RdnaAnalyzer

from ml_switcheroo.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.compiler.backends.rdna.macros import expand_conv2d, expand_linear

__all__ = [
  "Comment",
  "Directive",
  "Immediate",
  "Instruction",
  "Label",
  "LabelRef",
  "Memory",
  "Modifier",
  "Operand",
  "RdnaNode",
  "SGPR",
  "VGPR",
  "RdnaParser",
  "RdnaLexer",
  "Token",
  "TokenType",
  "RdnaLifter",
  "RdnaAnalyzer",
  "RdnaEmitter",
  "RdnaSynthesizer",
  "expand_conv2d",
  "expand_linear",
]
