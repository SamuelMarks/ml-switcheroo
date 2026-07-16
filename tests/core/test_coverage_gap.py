"""Auto-generated doc."""

import pytest
import libcst as cst
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.compiler.ir import LogicalNode


def test_conversion_result_has_errors():
  """Auto-generated doc."""
  from ml_switcheroo.core.conversion_result import ConversionResult

  res = ConversionResult(errors=["err"])
  assert res.has_errors
  res2 = ConversionResult()
  assert not res2.has_errors


def test_escape_hatch_fallback():
  """Auto-generated doc."""
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  node = cst.Name("x")
  res = EscapeHatch.mark_failure(node, "test fallback")
  assert res is node


def test_graph_extractor_coverage():
  """Auto-generated doc."""
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


def test_graph_optimizer_processed_ids():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  n2 = LogicalNode("n2", "B")
  LogicalGraph(nodes=[n1, n2], edges=[LogicalEdge("n1", "n2"), LogicalEdge("n1", "n2")])
  opt._match_sequence(n1, ["A", "B"], {"n1": n1, "n2": n2}, {"n1": ["n2"]}, set())


def test_html_node_not_implemented():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.nodes import HtmlNode

  class DummyNode(HtmlNode):
    """Auto-generated doc."""

    pass

  with pytest.raises(NotImplementedError):
    DummyNode().to_html()


def test_latex_node_to_text():
  """Auto-generated doc."""
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyNode(LatexNode):
    """Auto-generated doc."""

    def to_latex(self):
      """Auto-generated doc."""
      return super().to_latex()

  assert DummyNode().to_latex() is None


def test_mlir_dialect_validate_false():
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.dialect import OpSchema
  from ml_switcheroo.core.mlir.nodes import OperationNode

  schema = OpSchema(name="foo", num_regions=1)
  op = OperationNode(name="bar")
  assert not schema.validate(op)


def test_mlir_gen_base_coverage():
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
  from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode

  mixin = BaseGeneratorMixin()
  op = OperationNode(name="test", attributes=[AttributeNode(name="foo", value=["a", "b"])])
  assert mixin._get_attr(op, "foo") == "[a, b]"
  assert mixin._create_dotted_name("").value == "unknown"


def test_mlir_node_to_text():
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.nodes import MlirNode

  class DummyNode(MlirNode):
    """Auto-generated doc."""

    def to_text(self):
      """Auto-generated doc."""
      return super().to_text()

  assert DummyNode().to_text() is None


def test_rewriter_interface():
  """Auto-generated doc."""
  from ml_switcheroo.core.rewriter.interface import RewriterPass

  class DummyPass(RewriterPass):
    """Auto-generated doc."""

    def transform(self, module, context):
      """Auto-generated doc."""
      return super().transform(module, context)

  assert DummyPass().transform(None, None) is None


def test_patcher_coverage():
  """Auto-generated doc."""
  from ml_switcheroo.core.rewriter.patcher import GraphPatcher, PatchAction
  from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
  import libcst as cst

  node = cst.Name("test")
  action = PatchAction(node_id="n1")
  patcher = GraphPatcher([action], {"n1": node}, PythonSnippetEmitter())
  assert patcher._handle_node(node, node) is node

  stmt = cst.SimpleStatementLine(body=[])
  assert (
    patcher._unwrap_stmt_if_nested(cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("y")), stmt)
    is stmt
  )
  stmt2 = cst.SimpleStatementLine(body=[cst.Expr(cst.Name("y"))])
  assert patcher._unwrap_stmt_if_nested(cst.Name("x"), stmt2) is stmt2


def test_tikz_nodes_coverage():
  """Auto-generated doc."""
  from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzNode, TikzGraph, TriviaNode

  class DummyNode(TikzBaseNode):
    """Auto-generated doc."""

    def to_text(self):
      """Auto-generated doc."""
      return super().to_text()

  assert DummyNode().to_text() is None

  tn = TikzNode("n1", 0.0, 0.0, "content", leading_trivia=[TriviaNode(" ")])
  assert " " in tn.to_text()

  tg = TikzGraph(options=[])
  assert "\\begin{tikzpicture}" in tg.to_text()


def test_tracer_coverage():
  """Auto-generated doc."""
  from ml_switcheroo.core.tracer import TraceLogger

  t = TraceLogger()
  t.end_phase()
  t.log_warning("test warning")
  assert any(e.type == "analysis_warning" for e in t._events)


def test_html_parser_edge_cases():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  # 1. Red box without ':'
  # 2. Empty attribute config
  # 3. No attributes (pass in init)
  # 4. Arg without '='
  # 5. Invalid expression for _safe_val fallback
  html = """
    <div class="box r">
        <span class="header-txt">MyLayer</span>
        <code></code>
    </div>
    <div class="box r">
        <span class="header-txt">layer2 : Linear</span>
        <code>args: x</code>
    </div>
    <div class="box b">
        <span class="header-txt">Conv</span>
        <code>invalid_arg_&&, padding=1</code>
    </div>
    """
  parser = HtmlParser(html)
  mod = parser.parse()
  assert mod is not None


def test_html_parser_empty_init():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = """
    <div class="box b">
        <span class="header-txt">Conv</span>
        <code></code>
    </div>
    """
  parser = HtmlParser(html)
  mod = parser.parse()
  assert mod is not None


def test_html_parser_more_edges():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = """
    Model: MyAwesomeModel
    <div class="box b">
        <span class="header-txt">Call (conv)</span>
        <code>args: x</code>
    </div>
    <div class="box b">
        <span class="header-txt">Call</span>
        <code></code>
    </div>
    """
  parser = HtmlParser(html)
  mod = parser.parse()
  assert "MyAwesomeModel" in mod.code


def test_html_create_call_no_config():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  call = parser._create_call("my.func")
  assert call is not None


def test_parse_args_empty():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  assert parser._parse_args_str("") == []


def test_html_parser_attr_with_config():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = """
    <div class="box r">
        <span class="header-txt">layer3 : Dense</span>
        <code>units=10</code>
    </div>
    """
  parser = HtmlParser(html)
  parser.parse()


def test_html_create_call_with_config():
  """Auto-generated doc."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  parser._create_call("my.func", "a=1")
