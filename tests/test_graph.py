"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor


def test_graph_extractor_init_pass():
  """Docstring."""
  code = """
class GeneratedNet:
    def __init__(self):
        self.conv = nn.Conv2d(1, 20, 5)
        self.fc = nn.Linear(20, 10, bias=True)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "GeneratedNet"
  assert "conv" in extractor.layer_registry
  assert "fc" in extractor.layer_registry
  assert extractor.layer_registry["conv"].kind == "Conv2d"
  assert extractor.layer_registry["fc"].metadata["bias"] == "True"


def test_graph_extractor_forward_pass():
  """Docstring."""
  code = """
class GeneratedNet:
    def __init__(self):
        self.conv = nn.Conv2d(1, 20, 5)

    def forward(self, x):
        y = self.conv(x)
        return y
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "GeneratedNet"
  assert "conv" in extractor.layer_registry
  assert len(extractor.graph.edges) == 2

  edge_sources = [e.source for e in extractor.graph.edges]
  edge_targets = [e.target for e in extractor.graph.edges]

  assert "Input_x" in edge_sources
  assert "conv" in edge_targets

  assert "conv" in edge_sources
  assert "output" in edge_targets


def test_graph_extractor_external_function():
  """Docstring."""
  code = """
class GeneratedNet:
    def forward(self, x):
        y = F.relu(x)
        return y
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert extractor.layer_registry["func_relu"].kind == "F.relu"


def test_graph_extractor_return_call():
  """Docstring."""
  code = """
class GeneratedNet:
    def forward(self, x):
        return self.fc(x)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "Input_x" in extractor.layer_registry
  edge_targets = [e.target for e in extractor.graph.edges]
  assert "output" in edge_targets


def test_graph_extractor_top_level_flow():
  """Docstring."""
  code = """
x = 1
y = F.relu(x)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry


def test_graph_extractor_top_level_expr():
  """Docstring."""
  code = """
F.relu(x)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry
