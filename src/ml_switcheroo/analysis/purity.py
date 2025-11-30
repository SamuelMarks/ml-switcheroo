"""
Static Purity Analysis for JAX Compliance.

This module provides the `PurityScanner`, a LibCST transformer that detects
operations unsafe for functional frameworks (like JAX). JAX transformation
(JIT, VMap, Grad) requires pure functions with no side effects.

Operations flagged:
1.  **I/O**: `print`, `input`, `open`, `write`.
2.  **Global State**: `global` keyword usage.
3.  **Closure State**: `nonlocal` keyword usage (Feature 05).
4.  **Structure Mutation**: List methods `append`, `extend`, `pop`, etc.
5.  **Global RNG**: Seeding operations like `random.seed`, `torch.manual_seed`.

Violations are marked via the `EscapeHatch` mechanism, wrapping the code
with warning comments without altering the logic.
"""

from typing import List, Optional, Set, Union
import libcst as cst

from ml_switcheroo.core.escape_hatch import EscapeHatch


class PurityScanner(cst.CSTTransformer):
  """
  Scans CST for impurities and wraps violations in EscapeHatch markers.

  This transformer operates at the statement level. If a statement contains
  unsafe expressions (like `print()` or `list.append()`), the entire statement
  is wrapped in failure markers with a specific reason.

  Attributes:
      _current_violations (List[str]): Accumulator of errors for the current statement.
      _IO_FUNCTIONS (Set[str]): Set of function names considered I/O side effects.
      _MUTATION_METHODS (Set[str]): Set of method names considered in-place mutation.
      _GLOBAL_RNG_METHODS (Set[str]): Set of method names that mutate global random state.
  """

  _IO_FUNCTIONS: Set[str] = {"print", "input", "open", "write"}

  _MUTATION_METHODS: Set[str] = {
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "sort",
    "reverse",
  }

  _GLOBAL_RNG_METHODS: Set[str] = {
    "seed",  # random.seed, numpy.random.seed
    "manual_seed",  # torch.manual_seed
    "set_seed",  # tensorflow.random.set_seed
  }

  def __init__(self):
    """Initializes the PurityScanner."""
    self._current_violations: List[str] = []

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """
    Enters a statement line. Resets violation tracking.

    Args:
        node: The statement line node being visited.

    Returns:
        True to verify children.
    """
    self._current_violations = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel]:
    """
    Exits a statement line.
    If violations were found within this statement, wraps it in the EscapeHatch.

    Args:
        original_node: The node before transformations.
        updated_node: The node after internal transformations.

    Returns:
        The original/updated node, potentially wrapped in an EscapeHatch if impure.
    """
    if self._current_violations:
      # Deduplicate reasons
      unique_reasons = sorted(list(set(self._current_violations)))
      reason_msg = f"Side-effect unsafe for JAX: {', '.join(unique_reasons)}"

      # We wrap the *updated_node*. Even if we don't change inner content,
      # using updated_node is standard practice in Transformers.
      return EscapeHatch.mark_failure(updated_node, reason_msg)

    return updated_node

  def visit_Global(self, node: cst.Global) -> Optional[bool]:
    """
    Detects `global` keyword usage.

    Global variables break functional purity assumptions.

    Args:
        node: The Global statement node.

    Returns:
        False to stop traversing children.
    """
    names = [n.name.value for n in node.names]
    self._current_violations.append(f"Global mutation ({', '.join(names)})")
    return False  # structural checking done

  def visit_Nonlocal(self, node: cst.Nonlocal) -> Optional[bool]:
    """
    Detects `nonlocal` keyword usage.

    Modifying closure variables (nonlocal state) breaks JAX JIT tracers which
    assume stateless functions or explicit argument passing.

    Args:
        node: The Nonlocal statement node.

    Returns:
        False to stop traversing children.
    """
    names = [n.name.value for n in node.names]
    self._current_violations.append(f"Nonlocal mutation ({', '.join(names)})")
    return False

  def visit_Call(self, node: cst.Call) -> Optional[bool]:
    """
    Inspects calls for I/O functions, list mutations, or global RNG seeding.

    Args:
        node: The Call node.

    Returns:
        True to traverse arguments.
    """
    # 1. Check Function Name (e.g., print())
    if isinstance(node.func, cst.Name):
      func_name = node.func.value
      if func_name in self._IO_FUNCTIONS:
        self._current_violations.append(f"I/O Call ({func_name})")

    # 2. Check Method Call (e.g., x.append(), random.seed())
    elif isinstance(node.func, cst.Attribute):
      attr_name = node.func.attr.value

      # 2a. List Mutation
      if attr_name in self._MUTATION_METHODS:
        self._current_violations.append(f"In-place Mutation (. {attr_name})")

      # 2b. Global Random Seeding
      elif attr_name in self._GLOBAL_RNG_METHODS:
        self._current_violations.append(f"Global RNG State (. {attr_name})")

      # 2c. Specific I/O (File objects)
      # Catch file.write() specifically if we missed 'write' in generic check
      elif attr_name == "write":
        self._current_violations.append("I/O Call (.write)")

    return True
