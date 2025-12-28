"""
Error Handling Mixin.

Manages the reporting of translation failures and the injection of Escape Hatch
mechanisms to preserve code validity when transformation is impossible.
"""

from typing import List, Union, Optional
import libcst as cst
from ml_switcheroo.core.escape_hatch import EscapeHatch


class ErrorHandlingMixin:
  """
  Mixin for bubbling errors and wrapping statements.

  Assumed attributes on self:
      _current_stmt_errors (List[str]): List of errors for current statement.
      _current_stmt_warnings (List[str]): List of warnings for current statement.
  """

  def _report_failure(self, reason: str) -> None:
    """
    Records a fatal translation error for the current statement.
    This will trigger the Escape Hatch wrapper.
    """
    self._current_stmt_errors.append(reason)

  def _report_warning(self, reason: str) -> None:
    """
    Records a non-fatal warning for the current statement.
    """
    self._current_stmt_warnings.append(reason)

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """
    Resets error tracking at the start of each line.
    Errors bubble up from children (Expressions) to this Statement handler.
    """
    self._current_stmt_errors = []
    self._current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel]:
    """
    Handles error bubbling from expression rewrites.

    If errors occurred during processing of this line's children,
    wrap the line in an ``EscapeHatch`` and revert to the original node.
    """
    if self._current_stmt_errors:
      unique_errors = list(dict.fromkeys(self._current_stmt_errors))
      message = "; ".join(unique_errors)
      # Revert to ORIGINAL node to ensure no partial mutations exist
      return EscapeHatch.mark_failure(original_node, message)

    if self._current_stmt_warnings:
      unique_warnings = list(dict.fromkeys(self._current_stmt_warnings))
      message = "; ".join(unique_warnings)
      # Warnings apply to UPDATED node
      return EscapeHatch.mark_failure(updated_node, message)

    return updated_node
