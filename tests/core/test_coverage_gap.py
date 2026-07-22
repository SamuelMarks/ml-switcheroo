"""Test suite for the Coverage Gap module."""

import pytest
import libcst as cst
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.compiler.ir import LogicalNode


def test_conversion_result_has_errors():
  """Verifies the behavior of conversion result has errors."""
  from ml_switcheroo.core.conversion_result import ConversionResult

  res = ConversionResult(errors=["err"])
  assert res.has_errors
  res2 = ConversionResult()
  assert not res2.has_errors


def test_escape_hatch_fallback():
  """Verifies the behavior of escape hatch fallback."""
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  node = cst.Name("x")
  res = EscapeHatch.mark_failure(node, "test fallback")
  assert res is node


def test_graph_extractor_coverage():
  """Verifies the behavior of graph extractor coverage."""
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
  """Verifies the behavior of graph optimizer processed ids."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  n2 = LogicalNode("n2", "B")
  LogicalGraph(nodes=[n1, n2], edges=[LogicalEdge("n1", "n2"), LogicalEdge("n1", "n2")])
  opt._match_sequence(n1, ["A", "B"], {"n1": n1, "n2": n2}, {"n1": ["n2"]}, set())


def test_html_node_not_implemented():
  """Verifies the behavior of HTML node not implemented."""
  from ml_switcheroo.core.html.nodes import HtmlNode

  class DummyNode(HtmlNode):
    """Dummy Node class for testing purposes."""

    pass

  with pytest.raises(NotImplementedError):
    DummyNode().to_html()


def test_latex_node_to_text():
  """Verifies the behavior of LaTeX node to text."""
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyNode(LatexNode):
    """Dummy Node class for testing purposes."""

    def to_latex(self):
      """Mock implementation of to LaTeX."""
      return super().to_latex()

  assert DummyNode().to_latex() is None


def test_mlir_dialect_validate_false():
  """Verifies the behavior of MLIR dialect validate false."""
  from ml_switcheroo.core.mlir.dialect import OpSchema
  from ml_switcheroo.core.mlir.nodes import OperationNode

  schema = OpSchema(name="foo", num_regions=1)
  op = OperationNode(name="bar")
  assert not schema.validate(op)


def test_mlir_gen_base_coverage():
  """Verifies the behavior of MLIR generation base coverage."""
  from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
  from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode

  mixin = BaseGeneratorMixin()
  op = OperationNode(name="test", attributes=[AttributeNode(name="foo", value=["a", "b"])])
  assert mixin._get_attr(op, "foo") == "[a, b]"
  assert mixin._create_dotted_name("").value == "unknown"


def test_mlir_node_to_text():
  """Verifies the behavior of MLIR node to text."""
  from ml_switcheroo.core.mlir.nodes import MlirNode

  class DummyNode(MlirNode):
    """Dummy Node class for testing purposes."""

    def to_text(self):
      """Mock implementation of to text."""
      return super().to_text()

  assert DummyNode().to_text() is None


def test_rewriter_interface():
  """Verifies the behavior of rewriter interface."""
  from ml_switcheroo.core.rewriter.interface import RewriterPass

  class DummyPass(RewriterPass):
    """Dummy Pass class for testing purposes."""

    def transform(self, module, context):
      """Mock implementation of transform."""
      return super().transform(module, context)

  assert DummyPass().transform(None, None) is None


def test_patcher_coverage():
  """Verifies the behavior of patcher coverage."""
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
  """Verifies the behavior of TikZ nodes coverage."""
  from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzNode, TikzGraph, TriviaNode

  class DummyNode(TikzBaseNode):
    """Dummy Node class for testing purposes."""

    def to_text(self):
      """Mock implementation of to text."""
      return super().to_text()

  assert DummyNode().to_text() is None
  tn = TikzNode("n1", 0.0, 0.0, "content", leading_trivia=[TriviaNode(" ")])
  assert " " in tn.to_text()
  tg = TikzGraph(options=[])
  assert "\\begin{tikzpicture}" in tg.to_text()


def test_tracer_coverage():
  """Verifies the behavior of tracer coverage."""
  from ml_switcheroo.core.tracer import TraceLogger

  t = TraceLogger()
  t.end_phase()
  t.log_warning("test warning")
  assert any((e.type == "analysis_warning" for e in t._events))


def test_html_parser_edge_cases():
  """Verifies the behavior of HTML parser edge cases."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = '\n    <div class="box r">\n        <span class="header-txt">MyLayer</span>\n        <code></code>\n    </div>\n    <div class="box r">\n        <span class="header-txt">layer2 : Linear</span>\n        <code>args: x</code>\n    </div>\n    <div class="box b">\n        <span class="header-txt">Conv</span>\n        <code>invalid_arg_&&, padding=1</code>\n    </div>\n    '
  parser = HtmlParser(html)
  mod = parser.parse()
  assert mod is not None


def test_html_parser_empty_init():
  """Verifies the behavior of HTML parser empty initialization."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = (
    '\n    <div class="box b">\n        <span class="header-txt">Conv</span>\n        <code></code>\n    </div>\n    '
  )
  parser = HtmlParser(html)
  mod = parser.parse()
  assert mod is not None


def test_html_parser_more_edges():
  """Verifies the behavior of HTML parser more edges."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = '\n    Model: MyAwesomeModel\n    <div class="box b">\n        <span class="header-txt">Call (conv)</span>\n        <code>args: x</code>\n    </div>\n    <div class="box b">\n        <span class="header-txt">Call</span>\n        <code></code>\n    </div>\n    '
  parser = HtmlParser(html)
  mod = parser.parse()
  assert "MyAwesomeModel" in mod.code


def test_html_create_call_no_config():
  """Verifies the behavior of HTML create call no configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  call = parser._create_call("my.func")
  assert call is not None


def test_parse_args_empty():
  """Parses arguments empty."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  assert parser._parse_args_str("") == []


def test_html_parser_attr_with_config():
  """Verifies the behavior of HTML parser attribute with configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  html = '\n    <div class="box r">\n        <span class="header-txt">layer3 : Dense</span>\n        <code>units=10</code>\n    </div>\n    '
  parser = HtmlParser(html)
  parser.parse()


def test_html_create_call_with_config():
  """Verifies the behavior of HTML create call with configuration."""
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  parser._create_call("my.func", "a=1")
