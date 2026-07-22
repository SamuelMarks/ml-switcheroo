"""Test suite for the Tikz Analyser module."""

import libcst as cst
from ml_switcheroo.core.tikz.analyser import GraphExtractor
from ml_switcheroo.core.compiler.ir import LogicalGraph


def parse_and_extract(code: str) -> LogicalGraph:
  """Parses and extract."""
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor.graph


def test_extract_nodes_from_init():
  """Extracts nodes from initialization."""
  code = "\nclass Net:\n    def __init__(self):\n        self.conv1 = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(128, 10)\n"
  graph = parse_and_extract(code)
  assert len(graph.nodes) == 2
  conv = next((n for n in graph.nodes if n.id == "conv1"))
  assert conv.kind == "Conv2d"
  assert conv.metadata["arg_0"] == "1"
  assert conv.metadata["arg_1"] == "32"
  assert conv.metadata["arg_2"] == "3"
  fc = next((n for n in graph.nodes if n.id == "fc"))
  assert fc.kind == "Linear"
  assert fc.metadata["arg_0"] == "128"


def test_extract_edges_sequential_flow():
  """Extracts edges sequential flow."""
  code = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv(1, 1)\n        self.fc = nn.Linear(1, 1)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = self.fc(x)\n        return x\n"
  graph = parse_and_extract(code)
  assert len(graph.edges) == 3
  e1 = graph.edges[0]
  assert e1.source == "input"
  assert e1.target == "conv"
  e2 = graph.edges[1]
  assert e2.source == "conv"
  assert e2.target == "fc"
  e3 = graph.edges[2]
  assert e3.source == "fc"
  assert e3.target == "output"


def test_functional_call_tracing():
  """Verifies the behavior of functional call tracing."""
  code = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv2d(1,1)\n\n    def forward(self, img):\n        y = self.conv(img)\n        z = F.relu(y)\n        return z\n"
  graph = parse_and_extract(code)
  node_ids = {n.id for n in graph.nodes}
  assert "conv" in node_ids
  relu_node_found = any(("func_relu" in nid for nid in node_ids))
  assert relu_node_found
  edge1 = next((e for e in graph.edges if e.target == "conv"))
  assert edge1.source == "input"
  relu_id = next((nid for nid in node_ids if "func_relu" in nid))
  edge2 = next((e for e in graph.edges if e.target == relu_id))
  assert edge2.source == "conv"


def test_keyword_argument_extraction():
  """Verifies the behavior of keyword argument extraction."""
  code = "\nclass Layer:\n    def __init__(self):\n        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n"
  graph = parse_and_extract(code)
  pool = next((n for n in graph.nodes if n.id == "pool"))
  assert pool.metadata["kernel_size"] == "2"
  assert pool.metadata["stride"] == "2"


def test_ignore_constants_reused():
  """Verifies the behavior of ignore constants reused."""
  code = "\nclass Model:\n    def __init__(self):\n        self.layer = Op()\n    def forward(self, x):\n        return self.layer(x, 1.0)\n"
  graph = parse_and_extract(code)
  assert len(graph.edges) >= 1
  edges = graph.edges
  assert edges[0].source == "input"
  assert edges[0].target == "layer"
  assert edges[1].source == "layer"
  assert edges[1].target == "output"
