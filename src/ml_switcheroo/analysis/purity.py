"""
Static Purity Analysis for JAX Compliance.

This module provides the `PurityScanner`, a LibCST transformer that detects
operations unsafe for functional frameworks (like JAX). JAX transformation
(JIT, VMap, Grad) requires pure functions with no side effects.

Operations flagged:
1.  **I/O**: `print`, `input`, `open`, `write` (Standard Python).
2.  **Global State**: `global` keyword usage.
3.  **Closure State**: `nonlocal` keyword usage (Feature 05).
4.  **Structure Mutation**: List methods `append`, `extend`, etc.
5.  **Global RNG**: Seeding operations (dynamically loaded from semantic config).
6.  **Framework Impurities**: Methods like `add_`, `copy_` loaded from source framework config.

Violations are marked via the `EscapeHatch` mechanism.
"""

from typing import List, Optional, Set, Union, Any
import libcst as cst

from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.schema import StructuralTraits


class PurityScanner(cst.CSTTransformer):
  """
  Scans CST for impurities and wraps violations in EscapeHatch markers.

  Attributes:
      _current_violations (List[str]): Accumulator of errors for the current statement.
      _IO_FUNCTIONS (Set[str]): Standard Python I/O function names.
      _MUTATION_METHODS (Set[str]): Standard Python container mutation methods.
      _dynamic_impurity_methods (Set[str]): Methods loaded from framework configs (e.g. `add_`).
      _global_rng_methods (Set[str]): Methods loaded from framework configs (e.g. `manual_seed`).
  """

  # Standard Python Language Impurities (Constant across frameworks)
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

  # Fallback only if semantics not provided
  _DEFAULT_RNG_METHODS: Set[str] = {"seed"}

  def __init__(self, semantics: Any = None, source_fw: Optional[str] = None):
    """
    Initializes the PurityScanner.

    Args:
        semantics: SemanticsManager instance to load dynamic configs.
        source_fw: The framework being analyzed (to load specific impure methods).
                   If None, framework-specific impurity checks are skipped.
    """
    self._current_violations: List[str] = []
    self.source_fw = source_fw

    # Initialize sets
    self._global_rng_methods = set(self._DEFAULT_RNG_METHODS)
    self._dynamic_impurity_methods = set()

    # Dynamic Loading
    if semantics:
      # A. Global RNG Methods
      if hasattr(semantics, "get_all_rng_methods"):
        self._global_rng_methods.update(semantics.get_all_rng_methods())

      # B. Source Framework Specific Impurities (e.g. add_, copy_)
      if source_fw and hasattr(semantics, "get_framework_config"):
        conf = semantics.get_framework_config(source_fw)
        if conf and "traits" in conf:
          traits = StructuralTraits.model_validate(conf["traits"])
          self._dynamic_impurity_methods.update(traits.impurity_methods)

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """
    Enters a statement line. Resets violation tracking.

    Args:
        node: The statement line node.

    Returns:
        Always True to visit children.
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
        original_node: The original CST node structure.
        updated_node: The potentially transformed inner node logic.

    Returns:
        The wrapped node if unsafe, otherwise the updated node.
    """
    if self._current_violations:
      # Deduplicate reasons
      unique_reasons = sorted(list(set(self._current_violations)))
      reason_msg = f"Side-effect unsafe for JAX: {', '.join(unique_reasons)}"

      # We wrap the *updated_node*.
      return EscapeHatch.mark_failure(updated_node, reason_msg)

    return updated_node

  def visit_Global(self, node: cst.Global) -> Optional[bool]:
    """
    Detects usage of the 'global' keyword.

    Args:
        node: The global statement node.

    Returns:
        False to stop recursion (impure).
    """
    names = [n.name.value for n in node.names]
    self._current_violations.append(f"Global mutation ({', '.join(names)})")
    return False

  def visit_Nonlocal(self, node: cst.Nonlocal) -> Optional[bool]:
    """
    Detects usage of the 'nonlocal' keyword.

    Args:
        node: The nonlocal statement node.

    Returns:
        False to stop recursion (impure).
    """
    names = [n.name.value for n in node.names]
    self._current_violations.append(f"Nonlocal mutation ({', '.join(names)})")
    return False

  def visit_Call(self, node: cst.Call) -> Optional[bool]:
    """
    Inspects calls for I/O functions, list mutations, or global RNG seeding.

    Args:
        node: The call expression node.

    Returns:
        True to continue traversal.
    """
    # 1. Check Function Name (e.g., print())
    if isinstance(node.func, cst.Name):
      func_name = node.func.value
      if func_name in self._IO_FUNCTIONS:
        self._current_violations.append(f"I/O Call ({func_name})")

    # 2. Check Method Call (e.g., x.append(), random.seed(), x.add_())
    elif isinstance(node.func, cst.Attribute):
      attr_name = node.func.attr.value

      # 2a. List Mutation
      if attr_name in self._MUTATION_METHODS:
        self._current_violations.append(f"In-place Mutation (. {attr_name})")

      # 2b. Global Random Seeding (Dynamic)
      elif attr_name in self._global_rng_methods:
        self._current_violations.append(f"Global RNG State (. {attr_name})")

      # 2c. Framework Specific Impurity (New)
      elif attr_name in self._dynamic_impurity_methods:
        self._current_violations.append(f"State Mutation (. {attr_name})")

      # 2d. Specific I/O (File objects)
      elif attr_name == "write":
        self._current_violations.append("I/O Call (.write)")

    return True
