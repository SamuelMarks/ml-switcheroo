"""Tests for the HTML Compiler Backend."""

from pydantic import JsonValue
from unittest.mock import patch

from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_html_backend_init() -> None:
  """Verify html backend init behavior."""
  backend = HtmlBackend()
  assert backend is not None


def test_html_backend_format_args() -> None:
  """Verify html backend format args behavior."""
  backend = HtmlBackend()
  metadata = {"arg_1": "val1", "other": "val2", "arg_3": "val3"}
  formatted = backend._format_args(metadata)
  assert formatted == "val1, other=val2, val3"


def test_html_backend_is_stateful() -> None:
  """Verify html backend is stateful behavior."""
  backend = HtmlBackend()

  node_input = LogicalNode(id="in", op_type="Input")
  assert backend._is_stateful(node_input) is False

  node_output = LogicalNode(id="out", op_type="Output")
  assert backend._is_stateful(node_output) is False

  node_func_id = LogicalNode(id="func_1", op_type="Linear")
  assert backend._is_stateful(node_func_id) is False

  node_func_kind = LogicalNode(id="n1", op_type="func_call")
  assert backend._is_stateful(node_func_kind) is False

  node_upper = LogicalNode(id="n2", op_type="Linear")
  assert backend._is_stateful(node_upper) is True

  node_lower = LogicalNode(id="n3", op_type="relu")
  assert backend._is_stateful(node_lower) is False


def test_html_backend_clean_kind() -> None:
  """Verify html backend clean kind behavior."""
  backend = HtmlBackend()
  assert backend._clean_kind("func_call") == "Call"
  assert backend._clean_kind("a.b.c.Relu") == "Relu"
  assert backend._clean_kind("relu") == "Relu"


def test_html_backend_create_arrow() -> None:
  """Verify html backend create arrow behavior."""
  backend = HtmlBackend()

  arrow_def = backend._create_arrow(1, 2, "def")
  assert arrow_def.style_class == "s-red"
  assert arrow_def.x2 == 60
  assert arrow_def.y2 == 80

  arrow_data = backend._create_arrow(1, 2, "data")
  assert arrow_data.style_class == "s-green"
  assert arrow_data.x2 == 60
  assert arrow_data.y2 == 0

  arrow_seq = backend._create_arrow(1, 3, "seq")
  assert arrow_seq.style_class == "s-blue"
  assert arrow_seq.y2 == 50 + (3 - 1 - 1) * 120

  arrow_unknown = backend._create_arrow(1, 2, "unknown")
  assert arrow_unknown.x2 == 0
  assert arrow_unknown.y2 == 0


def test_html_backend_layout_graph_empty() -> None:
  """Verify html backend layout graph empty behavior."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  graph.add_node(LogicalNode(id="in", op_type="Input"))
  graph.add_node(LogicalNode(id="out", op_type="Output"))

  boxes = backend._layout_graph(graph)
  assert boxes == []


def test_html_backend_layout_graph_nodes() -> None:
  """Verify html backend layout graph nodes behavior."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="test_graph")

  n1 = LogicalNode(id="n1", op_type="Linear", attributes={"arg_x": "1"})
  n2 = LogicalNode(id="n2", op_type="relu", attributes={"alpha": "0.1"})
  n3 = LogicalNode(id="n3", op_type="Linear")

  graph.nodes.update({n.id: n for n in [n1, n2, n3]})

  boxes = backend._layout_graph(graph)
  assert len(boxes) == 9
  assert boxes[-1].css_class == "circ"


def test_html_backend_compile() -> None:
  """Verify html backend compile behavior."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="MyModel")
  n1 = LogicalNode(id="n1", op_type="relu")
  graph.add_node(n1)

  html_str = backend.compile(graph)
  assert "MyModel" in html_str


def test_html_backend_compile_default_name() -> None:
  """Verify html backend compile default name behavior."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="GeneratedNet")
  n1 = LogicalNode(id="n1", op_type="relu")
  graph.add_node(n1)

  html_str = backend.compile(graph)
  assert "ConvNet" in html_str


def test_html_backend_compile_no_name() -> None:
  """Verify html backend compile no name behavior."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="")
  n1 = LogicalNode(id="n1", op_type="relu")
  graph.add_node(n1)

  html_str = backend.compile(graph)
  assert "ConvNet" in html_str


def test_html_backend_layout_graph_no_last_blue_row() -> None:
  """Verify html backend layout graph with a single stateless op."""
  backend = HtmlBackend()
  graph = LogicalGraph(name="test")

  n1 = LogicalNode("n1", "add")
  graph.add_node(n1)
  boxes = backend._layout_graph(graph)
  assert len(boxes) >= 2


def test_html_backend_layout_graph_break_loop() -> None:
  """Verify html backend layout graph when GridBox css_class is mutated."""
  import ml_switcheroo.core.compiler.backends.html as html_mod

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.add_node(n1)
  graph.add_node(n2)

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args: JsonValue, **kwargs: JsonValue) -> html_mod.GridBox:
    """Execute mock gridbox helper logic.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    box = orig_gridbox(*args, **kwargs)  # type: ignore[arg-type]
    if "box b" in box.css_class:
      box.css_class = "box g"
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    boxes = backend._layout_graph(graph)
    assert len(boxes) >= 2


def test_html_backend_layout_graph_row_mismatch() -> None:
  """Verify html backend layout graph when blue box row does not match current row."""
  import ml_switcheroo.core.compiler.backends.html as html_mod

  backend = HtmlBackend()
  graph = LogicalGraph(name="test")
  n1 = LogicalNode("n1", "add")
  n2 = LogicalNode("n2", "add")
  graph.add_node(n1)
  graph.add_node(n2)

  orig_gridbox = html_mod.GridBox

  def mock_gridbox(*args: JsonValue, **kwargs: JsonValue) -> html_mod.GridBox:
    """Execute mock gridbox helper logic.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    box = orig_gridbox(*args, **kwargs)  # type: ignore[arg-type]
    if kwargs.get("css_class") == "box b" and kwargs.get("row") == 2:
      box.row = 999
    return box

  with patch("ml_switcheroo.core.compiler.backends.html.GridBox", side_effect=mock_gridbox):
    boxes = backend._layout_graph(graph)
    assert len(boxes) >= 2


def test_html_backend_first_operation_no_blue_row() -> None:
  """Verify html backend compile on empty graph produces valid HTML."""
  graph = LogicalGraph(name="test")
  backend = HtmlBackend()
  res = backend.compile(graph)
  assert "<!DOCTYPE html>" in res


def test_html_backend_first_op_not_layer() -> None:
  """Verify html backend compile with a single stateful node."""
  graph = LogicalGraph(name="test")
  op = LogicalNode(id="regular_op", op_type="Regular")
  graph.add_node(op)
  backend = HtmlBackend()
  code = backend.compile(graph)
  assert "box b" in code


def test_html_backend_second_operation_no_blue_row() -> None:
  """Verify html backend compile with two connected nodes."""
  nodes = {
    "n1": LogicalNode(id="n1", op_type="UnknownOp", attributes={}),
    "n2": LogicalNode(id="n2", op_type="UnknownOp", attributes={}),
  }
  edges = [LogicalEdge(source="n1", target="n2")]
  graph = LogicalGraph(name="test", nodes=nodes, edges=edges)

  backend = HtmlBackend()
  html_output = backend.compile(graph)
  assert "Unknownop" in html_output
