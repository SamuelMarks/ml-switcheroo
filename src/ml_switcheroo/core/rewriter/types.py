"""
Type definitions for the Rewriter module.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class SignatureContext:
  """
  Tracks the state of the current function scope being visited.

  Attributes:
      existing_args (Set[str]): Names of arguments defined in the function signature.
      injected_args (List[Tuple[str, Optional[str]]]): Args to append (name, annotation).
      preamble_stmts (List[str]): Code statements to prepend to the function body.
      is_init (bool): True if this function is `__init__`.
      is_module_method (bool): True if this function belongs to a Neural Module.
  """

  existing_args: Set[str] = field(default_factory=set)
  injected_args: List[Tuple[str, Optional[str]]] = field(default_factory=list)
  preamble_stmts: List[str] = field(default_factory=list)
  is_init: bool = False
  is_module_method: bool = False
