"""Unit tests for the TikZ Python-AST `GraphExtractor` (`ml_switcheroo.core.tikz.analyser`)."""

import libcst as cst

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.tikz.analyser import GraphExtractor


def extract_graph(code: str) -> GraphExtractor:
  """Parse `code` with LibCST and run `GraphExtractor` over the module.

  Args:
      code: Python source code defining a model class.

  Returns:
      Populated `GraphExtractor` visitor instance.
  """
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor


def test_extract_nodes_from_init_and_setup() -> None:
  """Verify `__init__` and `setup` register `self.<layer>` calls with positional and keyword attributes."""
  code = """
class MyModel:
    CLASS_VAR = 1

    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.size = 10
        local_var = nn.Linear(128, 10)
        self.layer.sub = nn.Conv1d(4, 8)
        other.conv = nn.Conv2d(3, 16)

    def setup(self):
        self.fc = nn.Linear(128, 10)

    def helper(self):
        return
"""
  extractor = extract_graph(code)
  assert extractor.model_name == "MyModel"
  assert set(extractor.layer_registry.keys()) == {"conv1", "pool", "fc"}

  conv = extractor.graph.nodes["conv1"]
  assert conv.op_type == "Conv2d"
  assert conv.attributes == {"arg_0": "1", "arg_1": "32", "arg_2": "3"}

  pool = extractor.graph.nodes["pool"]
  assert pool.attributes == {"kernel_size": "2", "stride": "2"}

  fc = extractor.graph.nodes["fc"]
  assert fc.op_type == "Linear"
  assert fc.attributes["arg_0"] == "128"


def test_sequential_and_functional_dataflow_edges() -> None:
  """Verify `forward` traces sequential stateful calls, functional `F.*` calls, and constant arguments."""
  code = """
class Net:
    def __init__(self):
        self.layer1 = Linear(10, 10)
        self.layer2 = Linear(10, 10)

    def forward(self, x):
        y = self.layer1(x, 1.0)
        z = F.relu(y)
        w = z
        return z
"""
  graph: LogicalGraph = extract_graph(code).graph
  edges = [(e.source, e.target) for e in graph.edges]
  assert ("input", "layer1") in edges
  assert ("layer1", "func_relu") in edges
  assert ("func_relu", "output") in edges
  assert {"input", "layer1", "layer2", "func_relu", "output"} <= set(graph.nodes.keys())


def test_direct_return_call_and_multiple_entry_methods() -> None:
  """Verify `return self.layer(x)`, `__call__`, `call`, and repeated `Return` visits reuse `output`."""
  code = """
class Net:
    def __init__(self):
        self.layer1 = Linear(10, 10)

    def forward(self, x):
        a = b = self.layer1(x)
        self.output[0] = self.layer1(a)
        return self.layer1(b)

    def __call__(self, img):
        img = F.relu(img)
        return img

    def call(self, val):
        untracked = 5
        return untracked
"""
  extractor = extract_graph(code)
  edges = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("input", "layer1") in edges
  assert ("layer1", "output") in edges

  # Visiting Return again when "output" is already registered exercises the existing-output branches
  ret_call_stmt = cst.ensure_type(cst.parse_statement("return self.conv(x)"), cst.SimpleStatementLine)
  ret_call = cst.ensure_type(ret_call_stmt.body[0], cst.Return)
  extractor._in_forward = True
  extractor.layer_registry["conv"] = LogicalNode("conv", "Conv2d", attributes={})
  extractor.visit_Return(ret_call)
  extractor.visit_Return(ret_call)

  ret_var_stmt = cst.ensure_type(cst.parse_statement("return x"), cst.SimpleStatementLine)
  ret_var = cst.ensure_type(ret_var_stmt.body[0], cst.Return)
  extractor.provenance["x"] = "conv"
  extractor.visit_Return(ret_var)
  extractor.visit_Return(ret_var)


def test_unresolvable_calls_and_helper_fallbacks() -> None:
  """Verify dynamic/lambda call targets and helper fallbacks safely produce no spurious edges."""
  code = """
class Net:
    def forward(self, x):
        y = (lambda t: t)(x)
        z = x[0]()
        return (lambda: None)()
"""
  extractor = extract_graph(code)
  assert extractor.graph.edges == []

  empty_extractor = GraphExtractor()
  assert empty_extractor._get_var_name(cst.Integer("1")) is None
  assert empty_extractor._node_to_string(cst.Name("var_name")) == "var_name"
  assert empty_extractor._resolve_layer_or_func_name(cst.parse_expression("(lambda x: x)")) is None
  empty_extractor._analyze_call_expression(cst.Call(func=cst.List([])), ["y"])
  empty_extractor._finalize_graph()
  assert empty_extractor.graph.nodes == {}
  assert empty_extractor.graph.edges == []
