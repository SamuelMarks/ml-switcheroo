"""Test suite for the Tikz Analyser module."""

import typing

import libcst as cst

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph
from ml_switcheroo.core.tikz.analyser import GraphExtractor


def parse_and_extract(code: str) -> LogicalGraph:
  """Docstring."""
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor.graph


def test_extract_nodes_from_init() -> None:
  """Docstring."""
  code: str = "\nclass Net:\n    def __init__(self):\n        self.conv1 = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(128, 10)\n"
  graph: LogicalGraph = parse_and_extract(code)
  assert len(graph.nodes) == 2
  conv: typing.Any = next((n for n in graph.nodes if n.id == "conv1"))
  assert conv.kind == "Conv2d"
  assert conv.metadata["arg_0"] == "1"
  assert conv.metadata["arg_1"] == "32"
  assert conv.metadata["arg_2"] == "3"
  fc: typing.Any = next((n for n in graph.nodes if n.id == "fc"))
  assert fc.kind == "Linear"
  assert fc.metadata["arg_0"] == "128"


def test_extract_edges_sequential_flow() -> None:
  """Docstring."""
  code: str = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv(1, 1)\n        self.fc = nn.Linear(1, 1)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = self.fc(x)\n        return x\n"
  graph: LogicalGraph = parse_and_extract(code)
  assert len(graph.edges) == 3
  e1: LogicalEdge = graph.edges[0]
  assert e1.source == "input"
  assert e1.target == "conv"
  e2: LogicalEdge = graph.edges[1]
  assert e2.source == "conv"
  assert e2.target == "fc"
  e3: LogicalEdge = graph.edges[2]
  assert e3.source == "fc"
  assert e3.target == "output"


def test_functional_call_tracing() -> None:
  """Verifies the behavior of functional call tracing."""
  code: str = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv2d(1,1)\n\n    def forward(self, img):\n        y = self.conv(img)\n        z = F.relu(y)\n        return z\n"
  graph: LogicalGraph = parse_and_extract(code)
  node_ids: set[str] = {n.id for n in graph.nodes}
  assert "conv" in node_ids
  relu_node_found: bool = any(("func_relu" in nid for nid in node_ids))
  assert relu_node_found
  edge1: LogicalEdge = next((e for e in graph.edges if e.target == "conv"))
  assert edge1.source == "input"
  relu_id: str = next((nid for nid in node_ids if "func_relu" in nid))
  edge2: LogicalEdge = next((e for e in graph.edges if e.target == relu_id))
  assert edge2.source == "conv"


def test_keyword_argument_extraction() -> None:
  """Docstring."""
  code: str = "\nclass Layer:\n    def __init__(self):\n        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n"
  graph: LogicalGraph = parse_and_extract(code)
  pool: typing.Any = next((n for n in graph.nodes if n.id == "pool"))
  assert pool.metadata["kernel_size"] == "2"
  assert pool.metadata["stride"] == "2"


def test_ignore_constants_reused() -> None:
  """Verifies the behavior of ignore constants reused."""
  code: str = "\nclass Model:\n    def __init__(self):\n        self.layer = Op()\n    def forward(self, x):\n        return self.layer(x, 1.0)\n"
  graph: LogicalGraph = parse_and_extract(code)
  assert len(graph.edges) >= 1
  edges: list[LogicalEdge] = graph.edges
  assert edges[0].source == "input"
  assert edges[0].target == "layer"
  assert edges[1].source == "layer"
  assert edges[1].target == "output"
