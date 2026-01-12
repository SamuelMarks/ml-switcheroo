"""
Rewriter Package.

This package provides the core AST transformation logic via a composable pipeline.

Components:
- `RewriterPipeline`: Orchestrates the execution of sequence passes.
- `RewriterContext`: Shared state container (Symbol Table, Semantics).
- `RewriterPass`: Base interface for transformation logic.

Core Passes:
- `StructuralPass`: Handles class inheritance, method renaming, signature changes.
- `ApiPass`: Handles API mapping, argument pivoting, and macro expansion.
- `AuxiliaryPass`: Handles decorators and control flow safety.
"""

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.interface import RewriterPass

# Passes
from ml_switcheroo.core.rewriter.passes.structure import StructuralPass
from ml_switcheroo.core.rewriter.passes.api import ApiPass
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass

__all__ = [
  "RewriterContext",
  "RewriterPipeline",
  "RewriterPass",
  "StructuralPass",
  "ApiPass",
  "AuxiliaryPass",
]
