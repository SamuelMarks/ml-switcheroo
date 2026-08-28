"""Docstring."""

from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.html.nodes import SvgArrow


def test_html_backend_compile() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="conv1", kind="Conv2d", metadata={"arg_1": "3", "arg_2": "16", "kernel_size": "3"}),
      LogicalNode(id="func_relu", kind="func_relu", metadata={}),
    ],
    edges=[],
    name="TestModel",
  )
  backend: HtmlBackend = HtmlBackend()
  html_str: str = backend.compile(graph)
  assert "TestModel" in html_str
  assert "Conv2d" in html_str
  assert "Relu" in html_str


def test_html_backend_empty_graph() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[],
    edges=[],
  )
  backend: HtmlBackend = HtmlBackend()
  html_str: str = backend.compile(graph)
  assert html_str is not None


def test_html_backend_input_output_only() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="out1", kind="Output"),
    ],
    edges=[],
  )
  backend: HtmlBackend = HtmlBackend()
  html_str: str = backend.compile(graph)
  assert html_str is not None


def test_clean_kind() -> None:
  """Docstring."""
  backend: HtmlBackend = HtmlBackend()
  assert backend._clean_kind("func_add") == "Add"
  assert backend._clean_kind("torch.nn.functional.relu") == "Relu"
  assert backend._clean_kind("Conv2d") == "Conv2d"


def test_is_stateful() -> None:
  """Docstring."""
  backend: HtmlBackend = HtmlBackend()
  assert backend._is_stateful(LogicalNode(id="in1", kind="Input")) is False
  assert backend._is_stateful(LogicalNode(id="out1", kind="Output")) is False
  assert backend._is_stateful(LogicalNode(id="func_1", kind="Add")) is False
  assert backend._is_stateful(LogicalNode(id="1", kind="func_add")) is False
  assert backend._is_stateful(LogicalNode(id="2", kind="Conv2d")) is True
  assert backend._is_stateful(LogicalNode(id="3", kind="add")) is False


def test_create_arrow() -> None:
  """Docstring."""
  backend: HtmlBackend = HtmlBackend()
  a1: SvgArrow = backend._create_arrow(1, 2, "def")
  assert a1.style_class == "s-red"

  a2: SvgArrow = backend._create_arrow(1, 2, "data")
  assert a2.style_class == "s-green"

  a3: SvgArrow = backend._create_arrow(1, 3, "seq")
  assert a3.style_class == "s-blue"
  assert a3.y2 == 50 + (3 - 1 - 1) * 120  # row_delta=2, so 50 + 120 = 170.

  a4: SvgArrow = backend._create_arrow(1, 2, "unknown")
  assert a4.x1 == 0 and a4.y1 == 0 and a4.x2 == 0 and a4.y2 == 0
