"""
SASS Lifter (Shim).

This module aliases the unified Lifter from the compiler frontend package
to maintain backward compatibility with previous core tests.
"""

from ml_switcheroo.compiler.frontends.sass.lifter import SassLifter

__all__ = ["SassLifter"]
