"""
SASS Macro Expansion Logic.

Re-exports macros from the Compiler Backend implementation to ensure consistency
and object identity across the codebase, avoiding duplication issues in tests.
"""

from ml_switcheroo.compiler.backends.sass.macros import (
  RegisterAllocatorProtocol,
  expand_conv2d,
  expand_linear,
)

__all__ = ["RegisterAllocatorProtocol", "expand_conv2d", "expand_linear"]
