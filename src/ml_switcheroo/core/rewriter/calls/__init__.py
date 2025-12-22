"""
Calls Rewriter Package.

This package handles the transformation of function calls, assignment unwrapping,
and complex API dispatching logic (infix operators, conditional rules).
"""

from ml_switcheroo.core.rewriter.calls.mixer import CallMixin

__all__ = ["CallMixin"]
