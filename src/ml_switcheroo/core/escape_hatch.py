"""
Escape Hatch Mechanism for Untranslatable Code.

This module provides the `EscapeHatch` class, responsible for wrapping code
that cannot be deterministically transpiled with specific marker comments.
It ensures that broken or partial translations are not emitted silently.

Reliability Logic:
- START_MARKER: injected into the leading lines of the failed statement.
- END_MARKER: injected via a sentinel statement (Ellipsis) following the failure.
- Verbatim Preservation: The caller is responsible for passing the *original* node
  to avoid emitting partially mutated (broken) code.
"""

import libcst as cst
from typing import Union


class EscapeHatch:
  """
  Handles the "Pass-Through" Protocol.
  Wraps untranslatable nodes with standardized comment flags and valid Python syntax markers.
  """

  START_MARKER = "# <SWITCHEROO_FAILED_TO_TRANS>"
  END_MARKER = "# </SWITCHEROO_FAILED_TO_TRANS>"

  @staticmethod
  def mark_failure(node: cst.CSTNode, reason: str) -> Union[cst.CSTNode, cst.FlattenSentinel]:
    """
    Attaches warning comments to the node and appends an end marker.

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
        A FlattenSentinel containing the preserved node (with header)
        and a footer node (Ellipsis) containing the end marker.
    """
    # 1. Create Header Lines (Start Marker + Reason)
    header_lines = [
      cst.EmptyLine(comment=cst.Comment(EscapeHatch.START_MARKER)),
      cst.EmptyLine(comment=cst.Comment(f"# Reason: {reason}")),
    ]

    # 2. Attach Header to the Node
    # We try to preserve existing leading lines if possible.
    try:
      current_leading = getattr(node, "leading_lines", [])
      marked_node = node.with_changes(leading_lines=[*header_lines, *current_leading])
    except (TypeError, AttributeError):
      # Fallback: If node type doesn't support leading_lines, return raw node.
      # This is rare if 'node' is a Statement.
      return node

    # 3. Create Footer Node (End Marker)
    # We use an Ellipsis (...) statement as a semantic no-op placeholder
    # to carry the leading_lines containing the END_MARKER.
    # This works because FlattenSentinel inserts it as the *next* statement.
    footer_node = cst.SimpleStatementLine(
      body=[cst.Expr(value=cst.Ellipsis())],
      leading_lines=[cst.EmptyLine(comment=cst.Comment(EscapeHatch.END_MARKER))],
    )

    # 4. Return both as a flattened sequence
    return cst.FlattenSentinel([marked_node, footer_node])
