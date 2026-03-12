import pytest
import libcst as cst


def test_conversion_result_has_errors():
  from ml_switcheroo.core.conversion_result import ConversionResult

  res = ConversionResult(errors=["err"])
  assert res.has_errors
  res2 = ConversionResult()
  assert not res2.has_errors


def test_escape_hatch_fallback():
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  node = cst.Name("x")
  res = EscapeHatch.mark_failure(node, "test fallback")
  assert res is node


def test_graph_extractor_coverage():
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
  from ml_switcheroo.core.graph_optimizer import GraphOptimizer
  from ml_switcheroo.compiler.ir import LogicalNode, LogicalEdge, LogicalGraph

  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  n2 = LogicalNode("n2", "B")
  g = LogicalGraph(nodes=[n1, n2], edges=[LogicalEdge("n1", "n2"), LogicalEdge("n1", "n2")])
  res = opt._match_sequence(n1, ["A", "B"], {"n1": n1, "n2": n2}, {"n1": ["n2"]}, set())


def test_html_node_not_implemented():
  from ml_switcheroo.core.html.nodes import HtmlNode

  class DummyNode(HtmlNode):
    pass

  with pytest.raises(NotImplementedError):
    DummyNode().to_html()


def test_latex_node_to_text():
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyNode(LatexNode):
    def to_latex(self):
      return super().to_latex()

  assert DummyNode().to_latex() is None


def test_mlir_dialect_validate_false():
  from ml_switcheroo.core.mlir.dialect import OpSchema
  from ml_switcheroo.core.mlir.nodes import OperationNode

  schema = OpSchema(name="foo", num_regions=1)
  op = OperationNode(name="bar")
  assert not schema.validate(op)


def test_mlir_gen_base_coverage():
  from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
  from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode

  mixin = BaseGeneratorMixin()
  op = OperationNode(name="test", attributes=[AttributeNode(name="foo", value=["a", "b"])])
  assert mixin._get_attr(op, "foo") == "[a, b]"
  assert mixin._create_dotted_name("").value == "unknown"


def test_mlir_node_to_text():
  from ml_switcheroo.core.mlir.nodes import MlirNode

  class DummyNode(MlirNode):
    def to_text(self):
      return super().to_text()

  assert DummyNode().to_text() is None


def test_rewriter_interface():
  from ml_switcheroo.core.rewriter.interface import RewriterPass

  class DummyPass(RewriterPass):
    def transform(self, module, context):
      return super().transform(module, context)

  assert DummyPass().transform(None, None) is None


def test_patcher_coverage():
  from ml_switcheroo.core.rewriter.patcher import GraphPatcher, PatchAction
  from ml_switcheroo.compiler.backends.python_snippet import PythonSnippetEmitter
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
  from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzNode, TikzGraph, TriviaNode

  class DummyNode(TikzBaseNode):
    def to_text(self):
      return super().to_text()

  assert DummyNode().to_text() is None

  tn = TikzNode("n1", 0.0, 0.0, "content", leading_trivia=[TriviaNode(" ")])
  assert " " in tn.to_text()

  tg = TikzGraph(options=[])
  assert "\\begin{tikzpicture}" in tg.to_text()


def test_tracer_coverage():
  from ml_switcheroo.core.tracer import TraceLogger

  t = TraceLogger()
  t.end_phase()
  t.log_warning("test warning")
  assert any(e.type == "analysis_warning" for e in t._events)


def test_html_parser_edge_cases():
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
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  call = parser._create_call("my.func")
  assert call is not None


def test_parse_args_empty():
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  assert parser._parse_args_str("") == []


def test_html_parser_attr_with_config():
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
  from ml_switcheroo.core.html.parser import HtmlParser

  parser = HtmlParser("")
  parser._create_call("my.func", "a=1")


def test_latex_parser_edges():
  from ml_switcheroo.core.latex.parser import LatexParser

  parser = LatexParser("")

  # 108: _parse_arg_list empty
  assert parser._parse_arg_list("   ") == []

  # 124: _safe_value_node ellipsis
  import libcst as cst

  assert isinstance(parser._safe_value_node("..."), cst.Ellipsis)

  # 129-133: _safe_value_node fallback
  # Provide something that CST can't parse as an expression but could be a Name or just causes an exception
  # e.g., an invalid python syntax string
  with __import__("unittest.mock").mock.patch(
    "libcst.parse_expression", side_effect=cst.ParserSyntaxError("msg", lines=[], raw_line=0, raw_column=0)
  ):
    node = parser._safe_value_node("valid_id")
  assert isinstance(node, cst.Name)

  # 143: _create_call without dots
  call = parser._create_call("myfunc")
  assert isinstance(call.func, cst.Name)

  # 156: _create_call arg_ prefix
  call = parser._create_call("f", config={"arg_0": "x"})
  assert len(call.args) == 1
  assert call.args[0].keyword is None

  # 170-172: _create_call args_list with =
  call = parser._create_call("f", args_list=["kw=val"])
  assert call.args[0].keyword.value == "kw"

  # 217: fallback inside generate_module
  from ml_switcheroo.core.latex.nodes import LatexNode

  class DummyOp(LatexNode):
    def __init__(self):
      super().__init__()
      self.output_id = "out"
      self.node_id = "out"

    def to_latex(self):
      return ""

  cdef = parser._synthesize_class("Test", [], None, [DummyOp()], None)
  import libcst as cst

  mod = cst.Module(body=[cdef])
  assert "None" in mod.code


def test_mlir_naming_edges():
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()

  # Manually populate reserved/used to force collisions
  strategy._used_names["class"] = "class"
  strategy._used_names["_class"] = "_class"
  strategy._used_names["_class_0"] = "_class_0"

  # Hint '%class' becomes 'class'.
  # 'class' is a python keyword so it might be in _reserved anyway, or in _used_names.
  # Doesn't start with '_', so attempt = '_class'.
  # '_class' is in _used_names, so falls to indexed fallback.
  # prefix = '_class'. attempt = '_class_0'.
  # '_class_0' is in _used_names, hits count += 1.
  # try '_class_1'. Succeeds.

  name = strategy.register("%class", hint="%class")
  assert name == "_class_1"


def test_mlir_naming_line_123():
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  # "class" is a keyword, attempt will be "_class".
  # "_class" is valid and not used, so it hits line 123.
  name = strategy.register("%class", hint="%class")
  assert name == "_class"


def test_graph_optimizer_lines():
  from ml_switcheroo.core.graph_optimizer import GraphOptimizer
  from ml_switcheroo.compiler.ir import LogicalNode

  opt = GraphOptimizer([])

  n1 = LogicalNode("n1", "A")
  # line 215: empty sequence
  assert opt._match_sequence(n1, [], {}, {}, set()) is None

  # line 233: tgt in processed_ids
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.compiler.ir import LogicalNode
import libcst as cst
from ml_switcheroo.core.import_fixer.resolution import _QualNameScanner


def test_graph_opt():
  opt = GraphOptimizer([])
  n1 = LogicalNode("n1", "A")
  # line 215: empty sequence
  assert opt._match_sequence(n1, [], {}, {}, set()) is None

  # line 233: tgt in processed_ids
  n2 = LogicalNode("n2", "B")
  nmap = {"n1": n1, "n2": n2}
  edges = {"n1": ["n2"]}
  proc = {"n2"}
  assert opt._match_sequence(n1, ["A", "B"], nmap, edges, proc) is None


def test_usage_visitor():
  visitor = _QualNameScanner("foo.bar")

  # Visit attribute but error inside get_full_name
  # line 68-69 (exception in visit_attribute)
  node = cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar"))
  # get_full_name fails if the node is malformed.
  # Let's mock get_full_name
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.import_fixer.resolution.get_full_name", side_effect=Exception("mocked")
  ):
    visitor.found = False
    visitor.visit_Attribute(node)
    visitor.visit_Attribute(node)

  # line 76
  visitor = _QualNameScanner("foo")
  visitor.visit_Name(cst.Name("foo"))
  assert visitor.found is True


def test_mlir_generator_gaps():
  from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
  from ml_switcheroo.core.mlir.nodes import OperationNode, ValueNode, AttributeNode, RegionNode, BlockNode
  import libcst as cst

  # Fake module structure
  # We need to pass a ModuleNode, but we can just instantiate the generator directly.
  gen = MlirToPythonGenerator()

  # 232-235: sw.import and unknown
  import_op = OperationNode('"sw.import"', [], [])
  with __import__("unittest.mock").mock.patch.object(gen, "_convert_import", return_value=None):
    assert (
      gen._convert_statement_op(import_op) is None
    )  # Assuming StatementGeneratorMixin lacks this, wait it might have it?
  # Let's mock _convert_import if it exists or just test the dispatch
  unknown_op = OperationNode('"sw.unknown_xyz"', [], [])
  assert gen._convert_statement_op(unknown_op) is None

  # 251: _wrap_as_statement _is_void_call
  # _is_void_call looks for e.g. super().__init__()
  # Let's mock _is_void_call
  op = OperationNode('"sw.call"', [ValueNode("%0")], [])
  gen.usage_counts["%0"] = 1
  with __import__("unittest.mock").mock.patch.object(gen, "_is_void_call", return_value=True):
    res = gen._wrap_as_statement(op, cst.Name("foo"))
    assert isinstance(res.body[0], cst.Expr)

  # 264-266: sw.getattr
  op_get = OperationNode('"sw.getattr"', [ValueNode("%1")], [AttributeNode("name", '"foo_attr"')])
  gen.usage_counts["%1"] = 1

  def mock_get_attr(op, attr):
    if attr == "type":
      return None
    return '"foo_attr"'

  with __import__("unittest.mock").mock.patch.object(gen, "_get_attr", side_effect=mock_get_attr):
    res = gen._wrap_as_statement(op_get, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_foo_attr"

  # 270: sw.constant
  op_const = OperationNode('"sw.constant"', [ValueNode("%2")], [])
  gen.usage_counts["%2"] = 1
  res = gen._wrap_as_statement(op_const, cst.Name("foo"))
  assert res.body[0].targets[0].target.value == "_cst"

  # 126: with_changes and leading
  # we need to simulate _convert_block returning a SimpleStatementLine with leading lines
  # It requires a block
  block = BlockNode("^bb0", operations=[OperationNode('"sw.call"', [ValueNode("%3")], [])])
  from ml_switcheroo.core.mlir.nodes import TriviaNode

  block.operations[0].leading_trivia = [TriviaNode("// test")]
  gen.usage_counts["%3"] = 1
  # Mock deferred_expr so it evaluates as a statement
  with __import__("unittest.mock").mock.patch.object(gen, "_create_expression_from_op", return_value=cst.Name("test")):
    stmts = gen._convert_block(block)
    assert len(stmts) == 1
    assert len(stmts[0].leading_lines) == 1


def test_stablehlo_emitter_gaps():
  from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
  from ml_switcheroo.core.mlir.nodes import OperationNode

  # Needs a semantics mock
  class MockSemantics:
    def get_definition(self, name):
      if name == "missing_variant":
        return ("id", {"variants": {}})
      return None

  emitter = StableHloEmitter(MockSemantics())

  # 172: no type attr
  op = OperationNode('"sw.call"', [], [])
  emitter._resolve_sw_op(op)
  assert op.name == '"sw.call"'

  # 207: no stablehlo variant
  assert emitter._lookup_stablehlo_op("missing_variant") is None

  # 225: bool
  assert emitter._map_py_type_to_mlir("bool") == "i1"

  # 229: unknown
  assert emitter._map_py_type_to_mlir("custom_object") == "!sw.unknown"


def test_structure_pass_coverage_245():
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  import libcst as cst
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager

  class MockSuper:
    pass

  class FakePass(MockSuper, StructuralTransformer):
    def __init__(self):
      self.context = type("MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": None})()
      self._in_annotation = False

  p = FakePass()
  node = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))

  # Mock hasattr to force line 245
  import builtins

  original_hasattr = builtins.hasattr

  def mock_hasattr(obj, name):
    if name == "leave_Attribute" and isinstance(obj, super):
      return False
    return original_hasattr(obj, name)

  with __import__("unittest.mock").mock.patch("builtins.hasattr", side_effect=mock_hasattr):
    res = p.leave_Attribute(node, node)
    assert res is node


def test_tikz_analyser_edges():
  from ml_switcheroo.core.tikz.analyser import GraphExtractor
  import libcst as cst

  code = """
class MyModel:
    def __init__(self):
        # 148: target is not self.something
        x = nn.Conv2d()
        
        # 155: value is not a call
        self.attr = 42

    def forward(self, x):
        # 180: value is not a call
        y = x
        
        # 217: _analyze_call_expression without layer_name
        # 208: _resolve_layer_or_func_name returns None (e.g. call a complex expression)
        z = x[0]()
        
        return z
"""
  mod = cst.parse_module(code)
  analyser = GraphExtractor()
  mod.visit(analyser)
