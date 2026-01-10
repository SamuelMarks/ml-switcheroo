"""
Shim Layer for SASS components.

Re-exports the new compiler components to maintain backward compatibility
with existing Framework Adapters and tests.
"""

# Re-export Frontend components (AST)
from ml_switcheroo.compiler.frontends.sass.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Memory,
  Operand,
  Predicate,
  Register,
  SassNode,
)

# Re-export Parser/Lifter
from ml_switcheroo.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.compiler.frontends.sass.tokens import SassLexer, Token, TokenType
from ml_switcheroo.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.compiler.frontends.sass.analysis import SassAnalyzer

# Re-export Backend components
from ml_switcheroo.compiler.backends.sass.emitter import SassEmitter
from ml_switcheroo.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.compiler.backends.sass.macros import expand_conv2d, expand_linear

__all__ = [
  "Comment",
  "Directive",
  "Immediate",
  "Instruction",
  "Label",
  "Memory",
  "Operand",
  "Predicate",
  "Register",
  "SassNode",
  "SassParser",
  "SassLexer",
  "Token",
  "TokenType",
  "SassAnalyzer",
  "SassLifter",
  "SassEmitter",
  "SassSynthesizer",
  "expand_conv2d",
  "expand_linear",
]
