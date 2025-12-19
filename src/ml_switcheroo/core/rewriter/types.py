"""
Type definitions for the Rewriter module.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class SignatureContext:
  """
  Tracks the state of the current function scope being visited.

  Used by the StructureMixin to maintain context about arguments,
  injections, and initialization status during AST traversal.
  """

  existing_args: Set[str] = field(default_factory=set)
  injected_args: List[Tuple[str, Optional[str]]] = field(default_factory=list)
  preamble_stmts: List[str] = field(default_factory=list)
  is_init: bool = False
  is_module_method: bool = False
