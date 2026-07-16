"""Auto-generated doc."""

import libcst as cst
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.import_fixer.resolution import _QualNameScanner


def test_latex_parser_edges():
  """Auto-generated doc."""
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
    """Auto-generated doc."""

    def __init__(self):
      """Auto-generated doc."""
      super().__init__()
      self.output_id = "out"
      self.node_id = "out"

    def to_latex(self):
      """Auto-generated doc."""
      return ""

  cdef = parser._synthesize_class("Test", [], None, [DummyOp()], None)
  import libcst as cst

  mod = cst.Module(body=[cdef])
  assert "None" in mod.code


def test_mlir_naming_edges():
  """Auto-generated doc."""
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
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  strategy = NamingContext()
  # "class" is a keyword, attempt will be "_class".
  # "_class" is valid and not used, so it hits line 123.
  name = strategy.register("%class", hint="%class")
  assert name == "_class"


def test_graph_optimizer_lines():
  """Auto-generated doc."""
  from ml_switcheroo.core.graph_optimizer import GraphOptimizer
  from ml_switcheroo.core.compiler.ir import LogicalNode

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


def test_graph_opt():
  """Auto-generated doc."""
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
  """Auto-generated doc."""
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
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
  from ml_switcheroo.core.mlir.nodes import OperationNode, ValueNode, AttributeNode, BlockNode
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
    """Auto-generated doc."""
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
  """Auto-generated doc."""
  from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
  from ml_switcheroo.core.mlir.nodes import OperationNode

  # Needs a semantics mock
  class MockSemantics:
    """Auto-generated doc."""

    def get_definition(self, name):
      """Auto-generated doc."""
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
  """Auto-generated doc."""
  from ml_switcheroo.core.rewriter.passes.structure import StructuralTransformer
  import libcst as cst

  class MockSuper:
    """Auto-generated doc."""

    pass

  class FakePass(MockSuper, StructuralTransformer):
    """Auto-generated doc."""

    def __init__(self):
      """Auto-generated doc."""
      self.context = type("MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": None})()
      self._in_annotation = False

  p = FakePass()
  node = cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))

  # Mock hasattr to force line 245
  import builtins

  original_hasattr = builtins.hasattr

  def mock_hasattr(obj, name):
    """Auto-generated doc."""
    if name == "leave_Attribute" and isinstance(obj, super):
      return False
    return original_hasattr(obj, name)

  with __import__("unittest.mock").mock.patch("builtins.hasattr", side_effect=mock_hasattr):
    res = p.leave_Attribute(node, node)
    assert res is node


def test_tikz_analyser_edges():
  """Auto-generated doc."""
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
