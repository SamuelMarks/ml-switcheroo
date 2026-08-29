"""Test suite for the Coverage Gap More2 module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.import_fixer.resolution import _QualNameScanner


def test_latex_parser_edges() -> None:
  """Verifies the behavior of LaTeX parser edges."""
  from ml_switcheroo.core.latex.parser import LatexParser

  parser = LatexParser("")
  assert parser._parse_arg_list("   ") == []
  import libcst as cst

  assert isinstance(parser._safe_value_node("..."), cst.Ellipsis)
  with __import__("unittest.mock").mock.patch(
    "libcst.parse_expression", side_effect=cst.ParserSyntaxError("msg", lines=[], raw_line=0, raw_column=0)
  ):
    node: typing.Any = parser._safe_value_node("valid_id")
  assert isinstance(node, cst.Name)
  call: typing.Any = parser._create_call("myfunc")
  assert isinstance(call.func, cst.Name)
  call = parser._create_call("f", config={"arg_0": "x"})
  assert len(call.args) == 1
  assert call.args[0].keyword is None
  call = parser._create_call("f", args_list=["kw=val"])
  assert typing.cast(cst.Name, call.args[0].keyword).value == "kw"
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyOp(LatexNode):
    def __init__(self) -> None:
      """Initializes the DummyOp instance."""
      super().__init__()
      self.output_id = "out"
      self.node_id = "out"

    def to_latex(self) -> str:
      """Mock implementation of to LaTeX."""
      return ""

  cdef: typing.Any = parser._synthesize_class("Test", [], None, [DummyOp()], None)  # type: ignore
  import libcst as cst

  mod = cst.Module(body=[cdef])
  assert "None" in mod.code


def test_mlir_naming_edges() -> None:
  """Verifies the behavior of MLIR naming edges."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  strategy._used_names["class"] = "class"
  strategy._used_names["_class"] = "_class"
  strategy._used_names["_class_0"] = "_class_0"
  name: str = strategy.register("%class", hint="%class")
  assert name == "_class_1"


def test_mlir_naming_line_123() -> None:
  """Verifies the behavior of MLIR naming line 123."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  name: str = strategy.register("%class", hint="%class")
  assert name == "_class"


def test_graph_optimizer_lines() -> None:
  """Verifies the behavior of graph optimizer lines."""
  from ml_switcheroo.core.compiler.ir import LogicalNode
  from ml_switcheroo.core.graph_optimizer import GraphOptimizer

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  assert opt._match_sequence(n1, [], {}, {}, set()) is None
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


def test_graph_opt() -> None:
  """Verifies the behavior of graph option."""
  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  assert opt._match_sequence(n1, [], {}, {}, set()) is None
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


def test_usage_visitor() -> None:
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


def test_mlir_generator_gaps() -> None:
  """Verifies the behavior of MLIR generator gaps."""
  import libcst as cst

  from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, ValueNode
  from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator

  gen = MlirToPythonGenerator()
  import_op = OperationNode(name='"sw.import"', operands=[], results=[])
  with __import__("unittest.mock").mock.patch.object(gen, "_convert_import", return_value=None):
    assert gen._convert_statement_op(import_op) is None
  unknown_op = OperationNode(name='"sw.unknown_xyz"', operands=[], results=[])
  assert gen._convert_statement_op(unknown_op) is None
  op = OperationNode(name='"sw.call"', operands=[ValueNode(name="%0")], results=[])
  gen.usage_counts["%0"] = 1
  with __import__("unittest.mock").mock.patch.object(gen, "_is_void_call", return_value=True):
    res: typing.Any = gen._wrap_as_statement(op, cst.Name("foo"))
    assert isinstance(res.body[0], cst.Expr)
  op_get = OperationNode(
    name='"sw.getattr"',
    operands=[ValueNode(name="%1")],
    results=[ValueNode(name="%2")],
    attributes=[AttributeNode(name="name", value='"foo_attr"')],
  )
  gen.usage_counts["%2"] = 1

  def mock_get_attr(o: typing.Any, attr: str) -> typing.Optional[str]:
    if attr == "type":
      return None
    return '"foo_attr"'

  with __import__("unittest.mock").mock.patch.object(gen, "_get_attr", side_effect=mock_get_attr):
    res = gen._wrap_as_statement(op_get, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_foo_attr"  # type: ignore
  op_const = OperationNode(name='"sw.constant"', operands=[], results=[ValueNode(name="%3")])
  gen.usage_counts["%3"] = 1
  res = gen._wrap_as_statement(op_const, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_cst"  # type: ignore
  block = BlockNode(
    label="^bb0", operations=[OperationNode(name='"sw.call"', operands=[], results=[ValueNode(name="%4")])]
  )  # type: ignore
  from ml_switcheroo.core.mlir.cst import Trivia

  block.operations[0].leading_trivia = [Trivia("// test")]  # type: ignore
  gen.usage_counts["%4"] = 1
  with __import__("unittest.mock").mock.patch.object(gen, "_create_expression_from_op", return_value=cst.Name("test")):
    stmts: list[typing.Any] = gen._convert_block(block)
    assert len(stmts) == 1
    assert len(stmts[0].leading_lines) == 1


def test_stablehlo_emitter_gaps() -> None:
  """Verifies the behavior of StableHLO emitter gaps."""
  from ml_switcheroo.core.mlir.cst import OperationNode
  from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter

  class MockSemantics:
    def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
      """Mock implementation of get definition."""
      if name == "missing_variant":
        return ("id", {"variants": {}})
      return None

  emitter = StableHloEmitter(MockSemantics())  # type: ignore
  op = OperationNode(name='"sw.call"', operands=[], results=[])
  emitter._resolve_sw_op(op)
  assert op.name == '"sw.call"'
  assert emitter._lookup_stablehlo_op("missing_variant") is None
  assert emitter._map_py_type_to_mlir("bool") == "i1"
  assert emitter._map_py_type_to_mlir("custom_object") == "!sw.unknown"


def test_structure_pass_coverage_245() -> None:
  """Verifies the behavior of structure pass coverage 245."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer

  class MockSuper:
    pass

  class FakePass(MockSuper, StructuralTransformer):  # type: ignore
    def __init__(self) -> None:
      """Initializes the FakePass instance."""
      self.context = type("MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": None})()
      self._in_annotation = False

  p = FakePass()
  node = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))
  import builtins

  original_hasattr = builtins.hasattr

  def mock_hasattr(obj: typing.Any, name: str) -> bool:
    if name == "leave_Attribute" and isinstance(obj, super):
      return False
    return original_hasattr(obj, name)

  with __import__("unittest.mock").mock.patch("builtins.hasattr", side_effect=mock_hasattr):
    res: typing.Any = p.leave_Attribute(node, node)
    assert res is node


def test_tikz_analyser_edges() -> None:
  """Verifies the behavior of TikZ analyser edges."""
  import libcst as cst

  from ml_switcheroo.core.tikz.analyser import GraphExtractor

  code: str = "\nclass MyModel:\n    def __init__(self):\n        # 148: target is not self.something\n        x = nn.Conv2d()\n\n        # 155: value is not a call\n        self.attr = 42\n\n    def forward(self, x):\n        # 180: value is not a call\n        y = x\n\n        # 217: _analyze_call_expression without layer_name\n        # 208: _resolve_layer_or_func_name returns None (e.g. call a complex expression)\n        z = x[0]()\n\n        return z\n"
  mod = cst.parse_module(code)
  analyser = GraphExtractor()
  mod.visit(analyser)


# --- Merged from test_coverage_gap.py ---


def test_conversion_result_has_errors() -> None:
  """Verifies the behavior of conversion result has errors."""
  from ml_switcheroo.core.conversion_result import ConversionResult

  res = ConversionResult(errors=["err"])
  assert res.has_errors
  res2 = ConversionResult()
  assert not res2.has_errors


def test_escape_hatch_fallback() -> None:
  """Verifies the behavior of escape hatch fallback."""
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  node = cst.Name("x")
  res: typing.Any = EscapeHatch.mark_failure(node, "test fallback")
  assert res is node


def test_graph_extractor_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.graph import GraphExtractor

  extractor = GraphExtractor()
  extractor._in_init = True
  node1 = cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Call(func=cst.Name("foo")))
  extractor.visit_Assign(node1)
  node2 = cst.Assign(
    targets=[cst.AssignTarget(cst.Attribute(value=cst.Name("self"), attr=cst.Name("layer")))], value=cst.Name("foo")
  )
  extractor.visit_Assign(node2)
  node3 = cst.Assign(
    targets=[cst.AssignTarget(cst.Attribute(value=cst.Name("self"), attr=cst.Name("layer")))],
    value=cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"), keyword=cst.Name("kw"))]),
  )
  extractor.visit_Assign(node3)
  extractor._in_init = False
  extractor._in_forward = True
  extractor._scope_depth = 1
  node4 = cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.List([]))
  extractor.visit_Assign(node4)
  node5 = cst.Call(func=cst.List([]))
  extractor._analyze_call_expression(node5, [])


def test_graph_optimizer_processed_ids() -> None:
  """Verifies the behavior of graph optimizer processed ids."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  n2 = LogicalNode("n2", "B")
  LogicalGraph(nodes=[n1, n2], edges=[LogicalEdge("n1", "n2"), LogicalEdge("n1", "n2")])
  opt._match_sequence(n1, ["A", "B"], {"n1": n1, "n2": n2}, {"n1": ["n2"]}, set())


def test_html_node_not_implemented() -> None:
  """Verifies the behavior of HTML node not implemented."""
  from ml_switcheroo.core.html.nodes import HtmlNode

  class DummyNode(HtmlNode):
    pass

  with pytest.raises(NotImplementedError):
    DummyNode().to_html()


def test_latex_node_to_text() -> None:
  """Verifies the behavior of LaTeX node to text."""
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyNode(LatexNode):
    def to_latex(self) -> str:
      """Mock implementation of to LaTeX."""
      return super().to_latex()

  assert DummyNode().to_latex() == ""


def test_mlir_dialect_validate_false() -> None:
  """Verifies the behavior of MLIR dialect validate false."""
  from ml_switcheroo.core.mlir.cst import OperationNode
  from ml_switcheroo.core.mlir.dialect import OpSchema

  schema = OpSchema(name="foo", num_regions=1)
  op = OperationNode(name="bar")
  assert not schema.validate(op)


def test_mlir_gen_base_coverage() -> None:
  """Verifies the behavior of MLIR generation base coverage."""
  from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode
  from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin

  class MockGen(BaseGeneratorMixin):
    def map_op(self, op: OperationNode) -> typing.Any:
      pass

  mixin = MockGen()
  op = OperationNode(name="test", attributes=[AttributeNode(name="foo", value=["a", "b"])])
  assert mixin._get_attr(op, "foo") == "[a, b]"
  assert typing.cast(cst.Name, mixin._create_dotted_name("")).value == "unknown"


def test_mlir_node_to_text() -> None:
  """Verifies the behavior of MLIR node to text."""
  from ml_switcheroo.core.mlir.cst import MlirNode

  class DummyNode(MlirNode):
    def to_text(self) -> typing.Any:
      """Mock implementation of to text."""
      try:
        return super().to_text()
      except NotImplementedError:
        return None

  assert DummyNode().to_text() is None


def test_rewriter_interface() -> None:
  """Verifies the behavior of rewriter interface."""
  from ml_switcheroo.core.rewriter.interface import RewriterPass

  class DummyPass(RewriterPass):
    def transform(self, module: cst.Module, context: typing.Any) -> cst.Module:
      """Mock implementation of transform."""
      try:
        super().transform(module, context)
      except NotImplementedError:
        pass
      return module

  assert DummyPass().transform(cst.Module([]), None).code == "\n"


def test_patcher_coverage() -> None:
  """Verifies the behavior of patcher coverage."""
  import libcst as cst

  from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
  from ml_switcheroo.core.rewriter.patcher import DeleteAction, GraphPatcher

  node = cst.Name("test")
  action = DeleteAction(node_id="n1")
  patcher = GraphPatcher([action], {"n1": node}, PythonSnippetEmitter("mock"))
  assert patcher._handle_node(node, node) == cst.RemoveFromParent()
  stmt = cst.SimpleStatementLine(body=[cst.Pass()])
  res_unwrap: typing.Any = patcher._unwrap_stmt_if_nested(
    cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("y")), stmt
  )
  assert isinstance(res_unwrap, cst.FlattenSentinel)
  stmt2 = cst.SimpleStatementLine(body=[cst.Expr(cst.Name("y"))])
  assert patcher._unwrap_stmt_if_nested(cst.Name("x"), stmt2) is stmt2


def test_tikz_nodes_coverage() -> None:
  """Verifies the behavior of TikZ nodes coverage."""
  from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzGraph, TikzNode, TriviaNode

  class DummyNode(TikzBaseNode):
    def to_text(self) -> str:
      """Mock implementation of to text."""
      try:
        super().to_text()
      except NotImplementedError:
        pass
      return ""

  assert DummyNode().to_text() == ""
  tn = TikzNode("n1", 0.0, 0.0, "content", leading_trivia=[TriviaNode(" ")])  # type: ignore
  assert " " in tn.to_text()
  tg = TikzGraph(options=[])
  assert "\\begin{tikzpicture}" in tg.to_text()


def test_tracer_coverage() -> None:
  """Verifies the behavior of tracer coverage."""
  from ml_switcheroo.core.tracer import TraceLogger

  t = TraceLogger()
  t.end_phase()
  t.log_warning("test warning")
  assert any((e.type == "analysis_warning" for e in t._events))


def test_html_parser_edge_cases() -> None:
  """Verifies the behavior of HTML parser edge cases."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html: str = '\n    <div class="box r">\n        <span class="header-txt">MyLayer</span>\n        <code></code>\n    </div>\n    <div class="box r">\n        <span class="header-txt">layer2 : Linear</span>\n        <code>args: x</code>\n    </div>\n    <div class="box b">\n        <span class="header-txt">Conv</span>\n        <code>invalid_arg_&&, padding=1</code>\n    </div>\n    '
  parser = HtmlParser(html)
  mod: typing.Any = parser.parse()
  assert mod is not None


def test_html_parser_empty_init() -> None:
  """Verifies the behavior of HTML parser empty initialization."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html: str = (
    '\n    <div class="box b">\n        <span class="header-txt">Conv</span>\n        <code></code>\n    </div>\n    '
  )
  parser = HtmlParser(html)
  mod: typing.Any = parser.parse()
  assert mod is not None


def test_html_parser_more_edges() -> None:
  """Verifies the behavior of HTML parser more edges."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html: str = '\n    <h3>Model: MyAwesomeModel</h3>\n    <div class="box b">\n        <span class="header-txt">Call (conv)</span>\n        <code>args: x</code>\n    </div>\n    <div class="box b">\n        <span class="header-txt">Call</span>\n        <code></code>\n    </div>\n    '
  parser = HtmlParser(html)
  mod: typing.Any = parser.parse()
  assert "MyAwesomeModel" in mod.code


def test_html_create_call_no_config() -> None:
  """Verifies the behavior of HTML create call no configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  call: typing.Any = parser._create_call("my.func")
  assert call is not None


def test_parse_args_empty() -> None:
  """Parses arguments empty."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  assert parser._parse_args_str("") == []


def test_html_parser_attr_with_config() -> None:
  """Verifies the behavior of HTML parser attribute with configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html: str = '\n    <div class="box r">\n        <span class="header-txt">layer3 : Dense</span>\n        <code>units=10</code>\n    </div>\n    '
  parser = HtmlParser(html)
  parser.parse()


def test_html_create_call_with_config() -> None:
  """Verifies the behavior of HTML create call with configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  parser._create_call("my.func", "a=1")
