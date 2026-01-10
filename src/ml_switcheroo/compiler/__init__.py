"""
Compiler Package.

This package defines the core Intermediate Representation (IR) and Backend interfaces
for the ml-switcheroo compilation pipeline. It separates the logical definition
of a computation graph from the frontend parsing and backend generation logic.
"""

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge

__all__ = ["CompilerBackend", "LogicalGraph", "LogicalNode", "LogicalEdge"]
