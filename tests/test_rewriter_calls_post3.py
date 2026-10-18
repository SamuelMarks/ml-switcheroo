"""Test suite for rewriter calls post3."""

import typing


import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    pass


def test_missing_post_branch() -> None:
  """Verify missing post branch behavior."""
  rewriter: typing.Any = DummyRewriter()
  node: typing.Any = cst.Name("invalid")
  mapping: typing.Any = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping, "id")

  class RewriterNoReport:
    """Test helper class RewriterNoReport."""

    pass

  rewriter_no_report: typing.Any = RewriterNoReport()
  _res: typing.Any = handle_post_processing(rewriter_no_report, node, mapping, "id")
  assert _res is node
