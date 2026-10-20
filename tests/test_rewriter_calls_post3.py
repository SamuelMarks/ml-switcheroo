"""Test suite for rewriter calls post3."""

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    pass


def test_missing_post_branch() -> None:
  """Verify missing post branch behavior."""
  rewriter = DummyRewriter()
  node = cst.Name("invalid")
  mapping = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping, "id")  # type: ignore[arg-type]

  class RewriterNoReport:
    """Test helper class RewriterNoReport."""

    pass

  rewriter_no_report = RewriterNoReport()
  _res = handle_post_processing(rewriter_no_report, node, mapping, "id")  # type: ignore[arg-type]
  assert _res is node
