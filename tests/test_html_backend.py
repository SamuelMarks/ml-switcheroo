"""Test suite for html backend."""

from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


def test_html_backend_compile() -> None:
  """Verify html backend compile behavior."""
  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="conv1", op_type="Conv2d", attributes={"arg_1": "3", "arg_2": "16", "kernel_size": "3"}),
        LogicalNode(id="func_relu", op_type="func_relu", attributes={}),
      ]
    },
    edges=[],
    name="TestModel",
  )
  backend = HtmlBackend()
  html_str = backend.compile(graph)
  assert "TestModel" in html_str
  assert "Conv2d" in html_str
  assert "Relu" in html_str


def test_html_backend_empty_graph() -> None:
  """Verify html backend empty graph behavior."""
  graph = LogicalGraph(
    nodes={},
    edges=[],
  )
  backend = HtmlBackend()
  html_str = backend.compile(graph)
  assert html_str is not None


def test_html_backend_input_output_only() -> None:
  """Verify html backend input output only behavior."""
  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="out1", op_type="Output"),
      ]
    },
    edges=[],
  )
  backend = HtmlBackend()
  html_str = backend.compile(graph)
  assert html_str is not None


def test_clean_kind() -> None:
  """Verify clean kind behavior."""
  backend = HtmlBackend()
  assert backend._clean_kind("func_add") == "Add"
  assert backend._clean_kind("torch.nn.functional.relu") == "Relu"
  assert backend._clean_kind("Conv2d") == "Conv2d"


def test_is_stateful() -> None:
  """Verify is stateful behavior."""
  backend = HtmlBackend()
  assert backend._is_stateful(LogicalNode(id="in1", op_type="Input")) is False
  assert backend._is_stateful(LogicalNode(id="out1", op_type="Output")) is False
  assert backend._is_stateful(LogicalNode(id="func_1", op_type="Add")) is False
  assert backend._is_stateful(LogicalNode(id="1", op_type="func_add")) is False
  assert backend._is_stateful(LogicalNode(id="2", op_type="Conv2d")) is True
  assert backend._is_stateful(LogicalNode(id="3", op_type="add")) is False


def test_create_arrow() -> None:
  """Verify create arrow behavior."""
  backend = HtmlBackend()
  a1 = backend._create_arrow(1, 2, "def")
  assert a1.style_class == "s-red"

  a2 = backend._create_arrow(1, 2, "data")
  assert a2.style_class == "s-green"

  a3 = backend._create_arrow(1, 3, "seq")
  assert a3.style_class == "s-blue"
  assert a3.y2 == 50 + (3 - 1 - 1) * 120  # row_delta=2, so 50 + 120 = 170.

  a4 = backend._create_arrow(1, 2, "unknown")
  assert a4.x1 == 0 and a4.y1 == 0 and a4.x2 == 0 and a4.y2 == 0
