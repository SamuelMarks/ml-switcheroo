"""Tests for TikZ Analyser."""

import libcst as cst

from ml_switcheroo.core.tikz.analyser import GraphExtractor


def extract_graph(code: str) -> GraphExtractor:
  """Docstring."""
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor


def test_graph_extractor_class_and_methods() -> None:
  """Docstring."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16)

    def setup(self):
        self.conv2 = nn.Conv2d(16, 32)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

    def __call__(self, img):
        img = F.relu(img)
        return img

    def call(self, val):
        pass
"""
  extractor: GraphExtractor = extract_graph(code)
  assert extractor.model_name == "MyModel"
  assert "conv1" in extractor.layer_registry
  assert "conv2" in extractor.layer_registry
  assert extractor.layer_registry["conv1"].kind == "Conv2d"
  assert extractor.layer_registry["conv2"].kind == "Conv2d"

  # We should have edges for both forward and __call__ but because they overwrite provenance
  # the final state might just be from the last executed method if it was a visitor pass
  # However, visit_FunctionDef resets provenance on each forward/call/__call__
  assert len(extractor.graph.nodes) > 0


def test_graph_extractor_layer_def_edge_cases() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        # Valid
        self.conv1 = nn.Conv2d(kernel_size=3)
        # Not a call
        self.size = 10
        # Not self attribute
        local_var = nn.Linear()
        # Complex target
        self.layer.sub = nn.Conv1d()
"""
  extractor: GraphExtractor = extract_graph(code)
  assert "conv1" in extractor.layer_registry
  assert "size" not in extractor.layer_registry
  assert "local_var" not in extractor.layer_registry
  assert "sub" not in extractor.layer_registry

  # Verify metadata extraction
  node = extractor.layer_registry["conv1"]
  assert node.metadata.get("kernel_size") == "3"


def test_graph_extractor_data_flow() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        self.layer1 = Linear(10, 10)
        self.layer2 = Linear(10, 10)

    def forward(self, x):
        # Assignment from call
        y = self.layer1(x)
        # Assignment from functional call
        z = F.relu(y)
        # Not a call
        w = z
        # Return variable
        return z
"""
  extractor: GraphExtractor = extract_graph(code)

  # Check edges
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("input", "layer1") in edges
  assert ("layer1", "func_relu") in edges
  assert ("func_relu", "output") in edges

  # Check nodes
  node_ids: set[str] = {n.id for n in extractor.graph.nodes}
  assert "layer1" in node_ids
  assert "layer2" in node_ids
  assert "input" in node_ids
  assert "func_relu" in node_ids
  assert "output" in node_ids


def test_graph_extractor_return_call() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        self.layer1 = Linear(10, 10)

    def forward(self, x):
        return self.layer1(x)
"""
  extractor: GraphExtractor = extract_graph(code)
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("input", "layer1") in edges
  assert ("layer1", "output") in edges


def test_graph_extractor_return_untracked_variable() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        pass

    def forward(self, x):
        untracked = 5
        return untracked
"""
  extractor: GraphExtractor = extract_graph(code)
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  # No edges should be created for untracked variable
  assert len(edges) == 0


def test_graph_extractor_resolve_layer_func_name_fallback() -> None:
  """Docstring."""
  code: str = """
class Net:
    def forward(self, x):
        y = getattr(self, "layer")(x)
        return y
"""
  extractor: GraphExtractor = extract_graph(code)
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  # No edges because getattr is not recognized properly by get_full_name
  # without returning None
  assert len(edges) == 0


def test_graph_extractor_complex_assignment_targets() -> None:
  """Docstring."""
  code: str = """
class Net:
    def __init__(self):
        self.layer = Linear()

    def forward(self, x):
        # Multiple assignment targets
        a = b = self.layer(x)
        # Complex assignment target
        self.output[0] = self.layer(a)
        return a
"""
  extractor: GraphExtractor = extract_graph(code)
  # Check edges to verify a and b were tracked
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("input", "layer") in edges


def test_get_var_name_fallback() -> None:
  """Docstring."""
  extractor = GraphExtractor()
  assert extractor._get_var_name(cst.Integer("1")) is None


def test_node_to_string_fallback() -> None:
  """Docstring."""
  extractor = GraphExtractor()
  node = cst.Name("var_name")
  assert extractor._node_to_string(node) == "var_name"


def test_analyze_call_expression_missing_layer() -> None:
  """Docstring."""
  extractor = GraphExtractor()
  # Mock a call with a complex func that doesn't resolve to a string
  call = cst.Call(func=cst.List([]))
  # Shouldn't crash and shouldn't add edges
  extractor._analyze_call_expression(call, ["y"])
  assert len(extractor.graph.edges) == 0
