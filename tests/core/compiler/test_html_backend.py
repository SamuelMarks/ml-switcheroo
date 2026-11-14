"""Unit tests for `HtmlBackend` (`ml_switcheroo.core.compiler.backends.html`)."""

from unittest.mock import patch

import ml_switcheroo.core.compiler.backends.html as html_mod
from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.html.nodes import GridBox


def test_html_backend_helpers_layout_and_compilation() -> None:
  """Verify `HtmlBackend` formatting, statefulness, arrows, grid layout, and full HTML compilation."""
  backend = HtmlBackend()
  assert (
    backend._format_args({"leading_trivia": "# skip", "arg_1": "val1", "other": "val2", "arg_3": "val3"})
    == "val1, other=val2, val3"
  )
  assert backend._clean_kind("func_call") == "Call"
  assert backend._clean_kind("a.b.c.Relu") == "Relu"
  assert backend._clean_kind("relu") == "Relu"

  assert backend._is_stateful(LogicalNode(id="in", op_type="Input")) is False
  assert backend._is_stateful(LogicalNode(id="out", op_type="Output")) is False
  assert backend._is_stateful(LogicalNode(id="func_1", op_type="Linear")) is False
  assert backend._is_stateful(LogicalNode(id="n1", op_type="func_call")) is False
  assert backend._is_stateful(LogicalNode(id="n2", op_type="Linear")) is True
  assert backend._is_stateful(LogicalNode(id="n3", op_type="relu")) is False

  arrow_def = backend._create_arrow(1, 2, "def")
  assert (arrow_def.style_class, arrow_def.x2, arrow_def.y2) == ("s-red", 60, 80)
  arrow_data = backend._create_arrow(1, 2, "data")
  assert (arrow_data.style_class, arrow_data.x2, arrow_data.y2) == ("s-green", 60, 0)
  arrow_seq = backend._create_arrow(1, 3, "seq")
  assert (arrow_seq.style_class, arrow_seq.y2) == ("s-blue", 170)
  arrow_unk = backend._create_arrow(1, 2, "unknown")
  assert (arrow_unk.x1, arrow_unk.y1, arrow_unk.x2, arrow_unk.y2) == (0, 0, 0, 0)

  # Empty / Input-Output-only graphs
  empty_io = LogicalGraph(
    name="test",
    nodes={"in": LogicalNode("in", "Input"), "out": LogicalNode("out", "Output")},
  )
  assert backend._layout_graph(empty_io) == []
  assert "<!DOCTYPE html>" in backend.compile(empty_io)
  assert "<!DOCTYPE html>" in backend.compile(LogicalGraph())

  # Populated mixed stateful/stateless graph and default `ConvNet` naming fallbacks
  rich = LogicalGraph(
    name="MyModel",
    nodes={
      "n1": LogicalNode("n1", "Linear", attributes={"arg_x": "1"}),
      "n2": LogicalNode("n2", "relu", attributes={"alpha": "0.1"}),
      "n3": LogicalNode("n3", "Linear"),
    },
    edges=[LogicalEdge("n1", "n2"), LogicalEdge("n2", "n3")],
  )
  boxes = backend._layout_graph(rich)
  assert len(boxes) == 9
  assert boxes[-1].css_class == "circ"
  compiled = backend.compile(rich)
  assert "MyModel" in compiled
  assert "Linear" in compiled

  for fallback_name in ("GeneratedNet", ""):
    g_fb = LogicalGraph(name=fallback_name, nodes={"n1": LogicalNode("n1", "relu")})
    assert "ConvNet" in backend.compile(g_fb)

  # Defensive reverse-scan branches when `GridBox` `css_class` or `row` is mutated
  two_add = LogicalGraph(
    name="test",
    nodes={"n1": LogicalNode("n1", "add"), "n2": LogicalNode("n2", "add")},
  )

  class _GreenBox(GridBox):
    """`GridBox` subclass mutating `'box b'` into `'box g'` upon initialization."""

    def model_post_init(self, _ctx: None) -> None:
      """Mutate `css_class` when `'box b'` is present.

      Args:
          _ctx: Pydantic post-init context.
      """
      if "box b" in self.css_class:
        self.css_class = "box g"

  class _MismatchBox(GridBox):
    """`GridBox` subclass mutating row-2 blue boxes to `row = 999`."""

    def model_post_init(self, _ctx: None) -> None:
      """Mutate `row` when `css_class == 'box b'` and `row == 2`.

      Args:
          _ctx: Pydantic post-init context.
      """
      if self.css_class == "box b" and self.row == 2:
        self.row = 999

  with patch.object(html_mod, "GridBox", _GreenBox):
    assert len(backend._layout_graph(two_add)) >= 2

  with patch.object(html_mod, "GridBox", _MismatchBox):
    assert len(backend._layout_graph(two_add)) >= 2
