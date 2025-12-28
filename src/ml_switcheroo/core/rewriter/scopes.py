"""
Scope Management Mixin.

Tracks the nesting of code scopes (Modules, Classes, Functions) and manages
the flagging of stateful variables (e.g. determining if a variable in `forward`
was defined as a Layer in `__init__`).
"""

from typing import List, Set


class ScopingMixin:
  """
  Mixin for managing variable scope stacks.

  Assumed attributes on self:
      _scope_stack (List[Set[str]]): The stack of active scopes.
  """

  def _enter_scope(self) -> None:
    """Push a new scope onto the stack (e.g. entering a class or function)."""
    self._scope_stack.append(set())

  def _exit_scope(self) -> None:
    """Pop the current scope from the stack."""
    if len(self._scope_stack) > 1:
      self._scope_stack.pop()

  def _mark_stateful(self, var_name: str) -> None:
    """
    Marks a variable name as stateful in the current scope.

    Used for tracking Neural Layers to determine if calls should be rewritten
    as stateful invocations (e.g. `layer.apply(...)` instead of `layer(...)`).

    Args:
        var_name: The variable identifier (e.g., 'self.conv1').
    """
    if self._scope_stack:
      self._scope_stack[-1].add(var_name)

  def _is_stateful(self, var_name: str) -> bool:
    """
    Checks if a variable is marked as stateful in any active scope.

    Traverses the scope stack from inner to outer.

    Args:
        var_name: The variable identifier.

    Returns:
        bool: True if the variable was previously marked as stateful.
    """
    for scope in reversed(self._scope_stack):
      if var_name in scope:
        return True
    return False
