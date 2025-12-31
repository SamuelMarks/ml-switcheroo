"""
Tests for Static Graph Extraction.

Verifies:
1.  Node Extraction: Correctly identifying layers in __init__.
2.  Metadata Extraction: Capturing layer arguments (kernel_size, etc).
3.  Edge Tracing: Correctly linking input -> layer -> output in forward.
4.  Functional Tracing: Handling F.relu() calls.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.tikz.analyser import GraphExtractor, LogicalNode, LogicalEdge


def parse_and_extract(code: str) -> "LogicalGraph":
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor.graph


def test_extract_nodes_from_init():
  """
  Scenario: Basic ConvNet init.
  """
  code = """
class Net:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(128, 10)
"""
  graph = parse_and_extract(code)

  assert len(graph.nodes) == 2
  # Verify Conv1
  conv = next(n for n in graph.nodes if n.id == "conv1")
  assert conv.kind == "Conv2d"
  assert conv.metadata["arg_0"] == "1"
  assert conv.metadata["arg_1"] == "32"
  assert conv.metadata["arg_2"] == "3"

  # Verify FC
  fc = next(n for n in graph.nodes if n.id == "fc")
  assert fc.kind == "Linear"
  assert fc.metadata["arg_0"] == "128"


def test_extract_edges_sequential_flow():
  """
  Scenario: x = conv(x) -> x = fc(x)
  """
  code = """
class Net:
    def __init__(self):
        self.conv = nn.Conv(1, 1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
"""
  graph = parse_and_extract(code)

  # Expected Edges: input -> conv -> fc -> output
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
  """
  Scenario: x = self.conv(x); x = F.relu(x)
  """
  code = """
class Net:
    def __init__(self):
        self.conv = nn.Conv2d(1,1)

    def forward(self, img):
        y = self.conv(img)
        z = F.relu(y)
        return z
"""
  graph = parse_and_extract(code)

  # Nodes: input, conv, func_relu, output
  node_ids = {n.id for n in graph.nodes}
  assert "conv" in node_ids
  # Heuristic name for functional call F.relu -> func_relu
  relu_node_found = any("func_relu" in nid for nid in node_ids)
  assert relu_node_found

  # Start Edge: Input -> Conv (using variable name 'img')
  edge1 = next(e for e in graph.edges if e.target == "conv")
  assert edge1.source == "input"

  # Middle Edge: Conv -> Relu
  # We find the node corresponding to relu
  relu_id = next(nid for nid in node_ids if "func_relu" in nid)
  edge2 = next(e for e in graph.edges if e.target == relu_id)
  assert edge2.source == "conv"


def test_keyword_argument_extraction():
  """
  Verify keyword args in init are captured.
  """
  code = """
class Layer:
    def __init__(self):
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
"""
  graph = parse_and_extract(code)
  pool = next(n for n in graph.nodes if n.id == "pool")
  assert pool.metadata["kernel_size"] == "2"
  assert pool.metadata["stride"] == "2"


def test_ignore_constants_reused():
  """
  Verify that passing constants (not tracked variables) doesn't create broken edges.
  """
  code = """
class Model:
    def __init__(self):
        self.layer = Op()
    def forward(self, x):
        return self.layer(x, 1.0)
"""
  graph = parse_and_extract(code)
  # Edge input -> layer should exist
  assert len(graph.edges) >= 1
  # Check return edge handling
  # input(x) -> layer -> Output
  edges = graph.edges
  assert edges[0].source == "input"
  assert edges[0].target == "layer"

  # Check link to output from layer (handled by visit_Return special case)
  assert edges[1].source == "layer"
  assert edges[1].target == "output"
