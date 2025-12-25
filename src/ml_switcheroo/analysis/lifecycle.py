"""
Lifecycle Analysis for Class Member Initialization.

This module provides the `InitializationTracker`, a static analysis tool that
verifies that class members used in the forward pass (inference) are properly
defined in the initialization phase (`__init__`).

This check helps detect:
1.  Dynamic attribute definition (valid in Python, often invalid in static graph compilation like JAX/XLA).
2.  Typos in member names between init and forward.
3.  Implicit state that might be lost during transpilation if not explicitly declared.
"""

from dataclasses import dataclass, field
from typing import List, Set
import libcst as cst


@dataclass
class _ClassContext:
  """
  Tracks state for the current class scope.
  """

  name: str
  initialized_members: Set[str] = field(default_factory=set)
  used_in_forward: Set[str] = field(default_factory=set)
  in_init: bool = False
  in_forward: bool = False


class InitializationTracker(cst.CSTVisitor):
  """
  Scans classes to ensure members used in forward are initialized in __init__.

  It maintains a stack of Class Contexts to handle nested class definitions correctly.
  """

  def __init__(self):
    """Initializes the tracker with empty state."""
    self.warnings: List[str] = []
    self._scope_stack: List[_ClassContext] = []

  def visit_ClassDef(self, node: cst.ClassDef) -> None:
    """
    Enters a class definition.
    Pushes a new Context onto the stack.
    """
    self._scope_stack.append(_ClassContext(name=node.name.value))

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """
    Exits a class definition and computes the difference between usages and inits.
    If discrepancies are found, they are recorded in `self.warnings`.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack.pop()

    # Check for members used but not initialized
    # We assume members starting with '_' are private/internal and might be ignored,
    # but stricter graph compilers might still care. For now, we check all.
    missing = ctx.used_in_forward - ctx.initialized_members

    if missing:
      sorted_missing = sorted(list(missing))
      msg = (
        f"Class '{ctx.name}': Members used in forward/call but not initialized in __init__: {', '.join(sorted_missing)}"
      )
      self.warnings.append(msg)

  def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Tracks entry into __init__ or forward/call methods.
    Sets context flags `in_init` or `in_forward`.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack[-1]
    func_name = node.name.value

    if func_name == "__init__":
      ctx.in_init = True
    elif func_name in ["forward", "__call__", "call"]:
      ctx.in_forward = True

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Exits function scope.
    Resets context flags.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack[-1]
    func_name = node.name.value

    if func_name == "__init__":
      ctx.in_init = False
    elif func_name in ["forward", "__call__", "call"]:
      ctx.in_forward = False

  def visit_Assign(self, node: cst.Assign) -> None:
    """
    Tracks assignments to `self.x` inside `__init__`.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack[-1]
    if ctx.in_init:
      # Check targets for self.attribute
      for target in node.targets:
        self._check_assignment_target(target.target, ctx)

  def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
    """
    Tracks annotated assignments (`self.x: int = ...`) inside `__init__`.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack[-1]
    if ctx.in_init:
      self._check_assignment_target(node.target, ctx)

  def visit_Attribute(self, node: cst.Attribute) -> None:
    """
    Tracks attribute access (`self.x`) inside `forward`.
    """
    if not self._scope_stack:
      return

    ctx = self._scope_stack[-1]

    # We only care about usage tracking in forward methods
    if ctx.in_forward:
      # Check if accessing 'self.something'
      # Note: We visit children, so 'self.layer.sublayer' will trigger twice?
      # 'self.layer' -> Attribute(value=Name(self), attr=layer)
      # 'self.layer.sublayer' -> Attribute(value=Attribute(self, layer), attr=sublayer)
      # We want the root member on self.

      # Simple check: is value 'self'?
      if self._is_self(node.value):
        member_name = node.attr.value
        ctx.used_in_forward.add(member_name)

  def _check_assignment_target(self, node: cst.BaseExpression, ctx: _ClassContext) -> None:
    """
    Helper to extract attribute name from assignment target.
    Recurses for tuple unpacking.

    Args:
        node: The target expression node.
        ctx: Current class context.
    """
    # We look for self.name
    if isinstance(node, cst.Attribute) and self._is_self(node.value):
      ctx.initialized_members.add(node.attr.value)
    # Handle tuple unpacking? (self.x, self.y) = (1, 2)
    elif isinstance(node, (cst.Tuple, cst.List)):
      for element in node.elements:
        # Recursively check element.value
        self._check_assignment_target(element.value, ctx)

  def _is_self(self, node: cst.BaseExpression) -> bool:
    """
    Checks if a node is the Name 'self'.
    """
    return isinstance(node, cst.Name) and node.value == "self"
