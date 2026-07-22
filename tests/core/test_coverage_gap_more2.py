"""Test suite for the Coverage Gap More2 module."""

import libcst as cst
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.import_fixer.resolution import _QualNameScanner


def test_latex_parser_edges():
  """Verifies the behavior of LaTeX parser edges."""
  from ml_switcheroo.core.latex.parser import LatexParser

  parser = LatexParser("")
  assert parser._parse_arg_list("   ") == []
  import libcst as cst

  assert isinstance(parser._safe_value_node("..."), cst.Ellipsis)
  with __import__("unittest.mock").mock.patch(
    "libcst.parse_expression", side_effect=cst.ParserSyntaxError("msg", lines=[], raw_line=0, raw_column=0)
  ):
    node = parser._safe_value_node("valid_id")
  assert isinstance(node, cst.Name)
  call = parser._create_call("myfunc")
  assert isinstance(call.func, cst.Name)
  call = parser._create_call("f", config={"arg_0": "x"})
  assert len(call.args) == 1
  assert call.args[0].keyword is None
  call = parser._create_call("f", args_list=["kw=val"])
  assert call.args[0].keyword.value == "kw"
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyOp(LatexNode):
    """Dummy Op class for testing purposes."""

    def __init__(self):
      """Initializes the DummyOp instance."""
      super().__init__()
      self.output_id = "out"
      self.node_id = "out"

    def to_latex(self):
      """Mock implementation of to LaTeX."""
      return ""

  cdef = parser._synthesize_class("Test", [], None, [DummyOp()], None)
  import libcst as cst

  mod = cst.Module(body=[cdef])
  assert "None" in mod.code


def test_mlir_naming_edges():
  """Verifies the behavior of MLIR naming edges."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  strategy._used_names["class"] = "class"
  strategy._used_names["_class"] = "_class"
  strategy._used_names["_class_0"] = "_class_0"
  name = strategy.register("%class", hint="%class")
  assert name == "_class_1"


def test_mlir_naming_line_123():
  """Verifies the behavior of MLIR naming line 123."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  name = strategy.register("%class", hint="%class")
  assert name == "_class"


def test_graph_optimizer_lines():
  """Verifies the behavior of graph optimizer lines."""
  from ml_switcheroo.core.graph_optimizer import GraphOptimizer
  from ml_switcheroo.core.compiler.ir import LogicalNode

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  assert opt._match_sequence(n1, [], {}, {}, set()) is None
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


def test_graph_opt():
  """Verifies the behavior of graph option."""
  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  assert opt._match_sequence(n1, [], {}, {}, set()) is None
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


def test_usage_visitor():
  """Verifies the behavior of usage visitor."""
  visitor = _QualNameScanner("foo.bar")
  node = cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar"))
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.import_fixer.resolution.get_full_name", side_effect=Exception("mocked")
  ):
    visitor.found = False
    visitor.visit_Attribute(node)
    visitor.visit_Attribute(node)
  visitor = _QualNameScanner("foo")
  visitor.visit_Name(cst.Name("foo"))
  assert visitor.found is True


def test_mlir_generator_gaps():
  """Verifies the behavior of MLIR generator gaps."""
  from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
  from ml_switcheroo.core.mlir.nodes import OperationNode, ValueNode, AttributeNode, BlockNode
  import libcst as cst

  gen = MlirToPythonGenerator()
  import_op = OperationNode('"sw.import"', [], [])
  with __import__("unittest.mock").mock.patch.object(gen, "_convert_import", return_value=None):
    assert gen._convert_statement_op(import_op) is None
  unknown_op = OperationNode('"sw.unknown_xyz"', [], [])
  assert gen._convert_statement_op(unknown_op) is None
  op = OperationNode('"sw.call"', [ValueNode("%0")], [])
  gen.usage_counts["%0"] = 1
  with __import__("unittest.mock").mock.patch.object(gen, "_is_void_call", return_value=True):
    res = gen._wrap_as_statement(op, cst.Name("foo"))
    assert isinstance(res.body[0], cst.Expr)
  op_get = OperationNode('"sw.getattr"', [ValueNode("%1")], [AttributeNode("name", '"foo_attr"')])
  gen.usage_counts["%1"] = 1

  def mock_get_attr(op, attr):
    """Provides a mock get attribute for testing."""
    if attr == "type":
      return None
    return '"foo_attr"'

  with __import__("unittest.mock").mock.patch.object(gen, "_get_attr", side_effect=mock_get_attr):
    res = gen._wrap_as_statement(op_get, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_foo_attr"
  op_const = OperationNode('"sw.constant"', [ValueNode("%2")], [])
  gen.usage_counts["%2"] = 1
  res = gen._wrap_as_statement(op_const, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_cst"
  block = BlockNode("^bb0", operations=[OperationNode('"sw.call"', [ValueNode("%3")], [])])
  from ml_switcheroo.core.mlir.nodes import TriviaNode

  block.operations[0].leading_trivia = [TriviaNode("// test")]
  gen.usage_counts["%3"] = 1
  with __import__("unittest.mock").mock.patch.object(gen, "_create_expression_from_op", return_value=cst.Name("test")):
    stmts = gen._convert_block(block)
    assert len(stmts) == 1
    assert len(stmts[0].leading_lines) == 1


def test_stablehlo_emitter_gaps():
  """Verifies the behavior of StableHLO emitter gaps."""
  from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
  from ml_switcheroo.core.mlir.nodes import OperationNode

  class MockSemantics:
    """Mock Semantics class for testing purposes."""

    def get_definition(self, name):
      """Mock implementation of get definition."""
      if name == "missing_variant":
        return ("id", {"variants": {}})
      return None

  emitter = StableHloEmitter(MockSemantics())
  op = OperationNode('"sw.call"', [], [])
  emitter._resolve_sw_op(op)
  assert op.name == '"sw.call"'
  assert emitter._lookup_stablehlo_op("missing_variant") is None
  assert emitter._map_py_type_to_mlir("bool") == "i1"
  assert emitter._map_py_type_to_mlir("custom_object") == "!sw.unknown"


def test_structure_pass_coverage_245():
  """Verifies the behavior of structure pass coverage 245."""
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  import libcst as cst

  class MockSuper:
    """Mock Super class for testing purposes."""

    pass

  class FakePass(MockSuper, StructuralTransformer):
    """Fake Pass class for testing purposes."""

    def __init__(self):
      """Initializes the FakePass instance."""
      self.context = type("MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": None})()
      self._in_annotation = False

  p = FakePass()
  node = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  import builtins

  original_hasattr = builtins.hasattr

  def mock_hasattr(obj, name):
    """Provides a mock hasattr for testing."""
    if name == "leave_Attribute" and isinstance(obj, super):
      return False
    return original_hasattr(obj, name)

  with __import__("unittest.mock").mock.patch("builtins.hasattr", side_effect=mock_hasattr):
    res = p.leave_Attribute(node, node)
    assert res is node


def test_tikz_analyser_edges():
  """Verifies the behavior of TikZ analyser edges."""
  from ml_switcheroo.core.tikz.analyser import GraphExtractor
  import libcst as cst

  code = "\nclass MyModel:\n    def __init__(self):\n        # 148: target is not self.something\n        x = nn.Conv2d()\n\n        # 155: value is not a call\n        self.attr = 42\n\n    def forward(self, x):\n        # 180: value is not a call\n        y = x\n\n        # 217: _analyze_call_expression without layer_name\n        # 208: _resolve_layer_or_func_name returns None (e.g. call a complex expression)\n        z = x[0]()\n\n        return z\n"
  mod = cst.parse_module(code)
  analyser = GraphExtractor()
  mod.visit(analyser)
