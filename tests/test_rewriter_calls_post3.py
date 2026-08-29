"""Module docstring."""

from typing import Any, Dict

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    pass


def test_missing_post_branch() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()
  node: cst.Name = cst.Name("invalid")
  mapping: Dict[str, Any] = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping, "id")

  class RewriterNoReport:
    pass

  rewriter_no_report: RewriterNoReport = RewriterNoReport()
  handle_post_processing(rewriter_no_report, node, mapping, "id")
