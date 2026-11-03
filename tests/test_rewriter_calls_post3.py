"""Test suite for rewriter calls post3."""

from tests.conftest import _test_getattr

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
  _test_getattr({"v": handle_post_processing}, "__getitem__")("v")(
    _test_getattr({"v": rewriter}, "__getitem__")("v"),
    _test_getattr({"v": node}, "__getitem__")("v"),
    _test_getattr({"v": mapping}, "__getitem__")("v"),
    _test_getattr({"v": "id"}, "__getitem__")("v"),
  )

  class RewriterNoReport:
    """Test helper class RewriterNoReport."""

    pass

  rewriter_no_report = RewriterNoReport()
  _res = _test_getattr(
    {
      "v": _test_getattr({"v": handle_post_processing}, "__getitem__")("v")(
        _test_getattr({"v": rewriter_no_report}, "__getitem__")("v"),
        _test_getattr({"v": node}, "__getitem__")("v"),
        _test_getattr({"v": mapping}, "__getitem__")("v"),
        _test_getattr({"v": "id"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert _res is node
