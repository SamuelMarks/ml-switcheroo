"""Escape Hatch Mechanism for Untranslatable Code.

This module provides the `EscapeHatch` class, responsible for wrapping code
that cannot be deterministically transpiled with specific marker comments.
It ensures that broken or partial translations are not emitted silently.

Reliability Logic:

- START_MARKER: injected into the leading lines of the failed statement.
- END_MARKER: injected via a sentinel statement (Ellipsis) following the failure.
- Verbatim Preservation: The caller is responsible for passing the *original* node
  to avoid emitting partially mutated (broken) code.
"""

import typing

import libcst as cst

T = typing.TypeVar("T", bound=cst.CSTNode)


class EscapeHatch:
  """Handle the "Pass-Through" Protocol.

  Wraps untranslatable nodes with standardized comment flags and valid Python syntax markers.

  Attributes:
      START_MARKER: Leading comment marker indicating start of untranslated block.
      END_MARKER: Trailing comment marker indicating end of untranslated block.
  """

  START_MARKER: str = "# <SWITCHEROO_FAILED_TO_TRANS>"
  END_MARKER: str = "# </SWITCHEROO_FAILED_TO_TRANS>"

  @staticmethod
  def mark_failure(node: T, reason: str) -> T | cst.FlattenSentinel[cst.BaseStatement]:
    """Attach warning comments to a statement node and append an end marker.

    Transformation:
        original_stmt()

    Becomes:
        # <SWITCHEROO_FAILED_TO_TRANS>
        # Reason: ...
        original_stmt()
        # </SWITCHEROO_FAILED_TO_TRANS>
        ...

    Args:
        node: The CST node (typically original_node) to preserve.
        reason: Human-readable explanation of the failure.

    Returns:
        A FlattenSentinel containing the preserved statement (with header)
        and a footer node (Ellipsis) containing the end marker, or ``node``
        unchanged if it is not a statement supporting ``leading_lines``.
    """
    if not isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
      return node

    header_lines = [
      cst.EmptyLine(comment=cst.Comment(EscapeHatch.START_MARKER)),
      cst.EmptyLine(comment=cst.Comment(f"# Reason: {reason}")),
    ]

    try:
      current_leading = list(node.leading_lines)
      marked_node: cst.BaseStatement = node.with_changes(leading_lines=[*header_lines, *current_leading])
    except (TypeError, AttributeError):
      return node

    footer_node = cst.SimpleStatementLine(
      body=[cst.Expr(value=cst.Ellipsis())],
      leading_lines=[cst.EmptyLine(comment=cst.Comment(EscapeHatch.END_MARKER))],
    )

    stmts: list[cst.BaseStatement] = [marked_node, footer_node]
    return cst.FlattenSentinel(stmts)
