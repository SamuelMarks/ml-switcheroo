"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    pass


def test_missing_post_branch():
  """Docstring."""
  rewriter = DummyRewriter()
  node = cst.Pass()
  mapping = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping, "id")

  class RewriterNoReport:
    pass

  rewriter_no_report = RewriterNoReport()
  handle_post_processing(rewriter_no_report, node, mapping, "id")
