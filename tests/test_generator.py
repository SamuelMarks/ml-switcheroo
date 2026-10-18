"""Test suite for generator."""

import typing

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, RegionNode, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def build_mock_op(
  name: typing.Any,
  results: typing.Any = [],
  operands: typing.Any = [],
  attributes: typing.Any = [],
  regions: typing.Any = [],
  trivia: typing.Any = [],
) -> typing.Any:
  """Execute build mock op helper logic.

  Args:
      name: Test argument or fixture.
      results: Test argument or fixture.
      operands: Test argument or fixture.
      attributes: Test argument or fixture.
      regions: Test argument or fixture.
      trivia: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return OperationNode(
    name=name,
    results=[ValueNode(name=r) for r in results],
    operands=[ValueNode(name=o) for o in operands],
    attributes=[AttributeNode(name=k, value=v) for k, v in attributes],
    regions=regions,
    leading_trivia=trivia,
  )


def test_generator_init() -> None:
  """Verify generator init behavior."""
  gen: typing.Any = MlirToPythonGenerator()
  assert gen.ctx is not None
  assert gen.usage_counts == {}
  assert gen.usage_consumers == {}
  assert gen.deferred_exprs == {}


def test_generator_scan_block_usage() -> None:
  """Verify generator scan block usage behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  # Op with operands and results
  op1: typing.Any = build_mock_op(name="sw.constant", results=["%0"], attributes=[("value", "42")])
  op2: typing.Any = build_mock_op(name="sw.op", operands=["%0"], results=["%1"], attributes=[("type", '"add"')])

  # Nested region
  op3: typing.Any = build_mock_op(
    name="sw.func",
    operands=[],
    regions=[
      RegionNode(blocks=[BlockNode(label="bb0", operations=[build_mock_op(name="sw.op", operands=["%1", "%0"])])])
    ],
  )

  block: typing.Any = BlockNode(label="main", operations=[op1, op2, op3])
  gen._scan_block_usage(block)

  assert gen.usage_counts["%0"] == 2
  assert gen.usage_counts["%1"] == 1
  assert getattr(gen.usage_consumers["%1"], "name", None) == "sw.op"


def test_convert_trivia() -> None:
  """Verify convert trivia behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  class MockTrivia:
    """Test helper class MockTrivia."""

    def __init__(self, content: typing.Any) -> None:
      """Initialize test helper instance.

      Args:
          content: Test argument or fixture.
      """
      self.content: typing.Any = content

  class MockTrivia2:
    """Test helper class MockTrivia2."""

    def __init__(self, text: typing.Any) -> None:
      """Initialize test helper instance.

      Args:
          text: Test argument or fixture.
      """
      self.text: typing.Any = text

  trivia_list: typing.Any = [
    MockTrivia("// a comment"),
    MockTrivia2("// another comment"),
    MockTrivia("% weird MLIR comment"),
    MockTrivia("   \n  "),  # Should be ignored
  ]

  res: typing.Any = gen._convert_trivia(trivia_list)
  assert len(res) == 3
  assert getattr(res[0].comment, "value", None) == "# a comment"
  assert getattr(res[1].comment, "value", None) == "# another comment"
  assert getattr(res[2].comment, "value", None) == "#% weird MLIR comment"


def test_generator_should_inline_expression() -> None:
  """Verify generator should inline expression behavior."""
  gen: typing.Any = MlirToPythonGenerator()
  gen.usage_counts["%used_const"] = 1
  gen.usage_counts["%unused_const"] = 0
  gen.usage_counts["%used_getattr"] = 1
  gen.usage_counts["%super_op"] = 1
  gen.usage_counts["%fused_op"] = 1
  gen.usage_counts["%multi_used_op"] = 2

  # 1. Constants
  op_const_used: typing.Any = build_mock_op("sw.constant", results=["%used_const"])
  assert gen._should_inline_expression(op_const_used, cst.Integer("42")) is True

  op_const_unused: typing.Any = build_mock_op("sw.constant", results=["%unused_const"])
  assert gen._should_inline_expression(op_const_unused, cst.Integer("42")) is False

  # Getattr
  op_getattr: typing.Any = build_mock_op("sw.getattr", results=["%used_getattr"])
  assert gen._should_inline_expression(op_getattr, cst.Name("attr")) is True

  # 2. Void/Super
  op_super: typing.Any = build_mock_op("sw.op", results=["%super_op"], attributes=[("type", '"super"')])
  assert gen._should_inline_expression(op_super, cst.Name("sup")) is True

  # 3. Statement fusion
  op_fused: typing.Any = build_mock_op("sw.op", results=["%fused_op"])
  gen.usage_consumers["%fused_op"] = build_mock_op("sw.setattr")
  assert gen._should_inline_expression(op_fused, cst.Name("fusion")) is True

  op_multi: typing.Any = build_mock_op("sw.op", results=["%multi_used_op"])
  gen.usage_consumers["%multi_used_op"] = build_mock_op("sw.setattr")
  assert gen._should_inline_expression(op_multi, cst.Name("multi")) is False  # usage > 1

  # No results
  op_no_res: typing.Any = build_mock_op("sw.op")
  assert gen._should_inline_expression(op_no_res, cst.Name("none")) is False


def test_resolve_operand() -> None:
  """Verify resolve operand behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  # Deferred
  gen.deferred_exprs["%def"] = cst.Integer("42")
  assert isinstance(gen._resolve_operand("%def"), cst.Integer)

  # Name lookup via ctx
  gen.ctx.register("%val", hint="val")
  res: typing.Any = gen._resolve_operand("%val")
  assert isinstance(res, cst.Name)
  assert res.value == "val"

  # Dotted name (e.g. from ctx lookup)
  gen.ctx._map["%dotted"] = "module.val"
  res_dot: typing.Any = gen._resolve_operand("%dotted")
  assert isinstance(res_dot, cst.Attribute)
  assert getattr(res_dot.value, "value", None) == "module"
  assert getattr(res_dot.attr, "value", None) == "val"


def test_create_expression_from_op() -> None:
  """Verify create expression from op behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  assert isinstance(gen._create_expression_from_op(build_mock_op("sw.constant")), cst.Integer)
  assert isinstance(gen._create_expression_from_op(build_mock_op("sw.getattr")), cst.Name)  # default to Name('error')
  assert isinstance(gen._create_expression_from_op(build_mock_op("sw.call")), cst.Call)
  assert isinstance(gen._create_expression_from_op(build_mock_op("sw.op", attributes=[("type", '"unknown"')])), cst.Call)
  assert gen._create_expression_from_op(build_mock_op("sw.unknown")) is None


def test_convert_statement_op() -> None:
  """Verify convert statement op behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  assert isinstance(gen._convert_statement_op(build_mock_op("sw.return")), cst.SimpleStatementLine)
  assert isinstance(gen._convert_statement_op(build_mock_op("sw.setattr")), cst.SimpleStatementLine)
  assert isinstance(gen._convert_statement_op(build_mock_op("sw.import")), cst.SimpleStatementLine)
  assert isinstance(gen._convert_statement_op(build_mock_op("sw.module")), cst.ClassDef)
  assert isinstance(gen._convert_statement_op(build_mock_op("sw.func")), cst.FunctionDef)
  assert gen._convert_statement_op(build_mock_op("sw.unknown")) is None


def test_wrap_as_statement() -> None:
  """Verify wrap as statement behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  expr: typing.Any = cst.Name("val")

  # Unused usage count
  gen.usage_counts["%unused"] = 0
  op_unused: typing.Any = build_mock_op("sw.op", results=["%unused"])
  stmt_unused: typing.Any = gen._wrap_as_statement(op_unused, expr)
  assert isinstance(stmt_unused, cst.SimpleStatementLine)
  assert isinstance(stmt_unused.body[0], cst.Expr)

  # Used
  gen.usage_counts["%used"] = 1
  op_used: typing.Any = build_mock_op("sw.op", results=["%used"])
  stmt_used: typing.Any = gen._wrap_as_statement(op_used, expr)
  assert isinstance(stmt_used, cst.SimpleStatementLine)
  assert isinstance(stmt_used.body[0], cst.Assign)
  assert getattr(stmt_used.body[0].targets[0].target, "value", None) == "_used"

  # Semantic hint extraction from type
  gen.usage_counts["%used_hint"] = 1
  op_used_hint: typing.Any = build_mock_op("sw.op", results=["%used_hint"], attributes=[("type", '"torch.flatten"')])
  stmt_used_hint: typing.Any = gen._wrap_as_statement(op_used_hint, expr)
  assert getattr(stmt_used_hint.body[0].targets[0].target, "value", None) == "_flatten"

  # Semantic hint from name (getattr)
  gen.usage_counts["%used_getattr"] = 1
  op_used_getattr: typing.Any = build_mock_op("sw.getattr", results=["%used_getattr"], attributes=[("name", '"my_attr"')])
  stmt_used_getattr: typing.Any = gen._wrap_as_statement(op_used_getattr, expr)
  assert getattr(stmt_used_getattr.body[0].targets[0].target, "value", None) == "_my_attr"  # NamingContext prepends _

  # Constant hint
  gen.usage_counts["%used_const"] = 1
  op_used_const: typing.Any = build_mock_op("sw.constant", results=["%used_const"])
  stmt_used_const: typing.Any = gen._wrap_as_statement(op_used_const, expr)
  assert getattr(stmt_used_const.body[0].targets[0].target, "value", None) == "_cst"

  # No results
  op_no_res: typing.Any = build_mock_op("sw.op")
  stmt_no_res: typing.Any = gen._wrap_as_statement(op_no_res, expr)
  assert isinstance(stmt_no_res, cst.SimpleStatementLine)
  assert isinstance(stmt_no_res.body[0], cst.Expr)


def test_is_void_call() -> None:
  """Verify is void call behavior."""
  gen: typing.Any = MlirToPythonGenerator()

  # super().__init__()
  super_call: typing.Any = cst.Call(func=cst.Name("super"))
  init_attr: typing.Any = cst.Attribute(value=super_call, attr=cst.Name("__init__"))
  void_call: typing.Any = cst.Call(func=init_attr)

  assert gen._is_void_call(void_call) is True

  # some_other_call()
  norm_call: typing.Any = cst.Call(func=cst.Name("my_func"))
  assert gen._is_void_call(norm_call) is False
  assert gen._is_void_call(cst.Name("val")) is False


def test_generate_integration() -> None:
  """Verify generate integration behavior."""

  # Put it all together to test logic flow in _convert_block and generate
  class MockTrivia:
    """Test helper class MockTrivia."""

    def __init__(self, content: typing.Any) -> None:
      """Initialize test helper instance.

      Args:
          content: Test argument or fixture.
      """
      self.content: typing.Any = content

  # Need to build realistic block
  op1: typing.Any = build_mock_op(
    "sw.constant", results=["%0"], attributes=[("value", "42")], trivia=[MockTrivia("// inline me")]
  )
  op2: typing.Any = build_mock_op(
    "sw.op", results=["%1"], operands=["%0"], attributes=[("type", '"add"')], trivia=[MockTrivia("// assign me")]
  )
  op3: typing.Any = build_mock_op("sw.return", operands=["%1"])
  op4: typing.Any = build_mock_op("sw.import", attributes=[("names", "['sys']"), ("aliases", "['sys']")])

  block: typing.Any = BlockNode(label="main", operations=[op1, op2, op3, op4])
  mod: typing.Any = ModuleNode(body=block)

  gen: typing.Any = MlirToPythonGenerator()
  cst_mod: typing.Any = gen.generate(mod)

  assert isinstance(cst_mod, cst.Module)
  assert len(cst_mod.body) == 2  # op1 inlines into op2, op2 inlines into op3 -> sw.return, sw.import

  # Check that op1 and op2 inlined into op3 (via resolve operand logic)
  ret_stmt: typing.Any = cst_mod.body[0]
  assert isinstance(ret_stmt.body[0], cst.Return)
  assert isinstance(ret_stmt.body[0].value, cst.Call)  # op2
  assert getattr(ret_stmt.body[0].value.args[0].value, "value", None) == "42"  # op1 inlined const


def test_generator_convert_block_uncovered() -> None:
  """Verify generator convert block uncovered behavior."""
  gen: typing.Any = MlirToPythonGenerator()
  gen.usage_counts["%unused"] = 0
  # Test op that is wrapped as statement and gets comments
  op1: typing.Any = build_mock_op(
    "sw.call", results=["%not_inline"], trivia=[type("Mock", (), {"text": "// test comment"})()]
  )
  gen.usage_counts["%not_inline"] = 2

  # Test op that produces no expression and returns a stmt, with comments
  op2: typing.Any = build_mock_op(
    "sw.return", operands=["%not_inline"], trivia=[type("Mock", (), {"text": "// return comment"})()]
  )

  # Test op that produces an expression but no comments
  op4: typing.Any = build_mock_op("sw.call", results=["%not_inline_no_comment"])
  gen.usage_counts["%not_inline_no_comment"] = 2

  block: typing.Any = BlockNode(arguments=[], operations=[op1, op2, op4])

  # Add dummy expr generator rule for sw.call
  gen.deferred_exprs["%not_inline"] = cst.Call(func=cst.Name("dummy"))
  gen.deferred_exprs["%not_inline_no_comment"] = cst.Call(func=cst.Name("dummy2"))
  stmts: typing.Any = gen._convert_block(block)
  assert len(stmts) == 3
  assert getattr(stmts[0].leading_lines[0].comment, "value", None) == "# test comment"
  assert getattr(stmts[1].leading_lines[0].comment, "value", None) == "# return comment"
  assert len(stmts[2].leading_lines) == 0

  # Test empty statement branch
  op3: typing.Any = build_mock_op("sw.ignored_op")
  block2: typing.Any = BlockNode(arguments=[], operations=[op3])
  stmts2: typing.Any = gen._convert_block(block2)
  assert len(stmts2) == 0


def test_generator_wrap_as_statement_uncovered() -> None:
  """Verify generator wrap as statement uncovered behavior."""
  gen: typing.Any = MlirToPythonGenerator()
  gen.usage_counts["%used_once"] = 1
  gen.usage_consumers["%used_once"] = build_mock_op("sw.call")

  # test should_inline_expression [203, 206]
  op_not_inline: typing.Any = build_mock_op("sw.op", results=["%used_once"])
  assert gen._should_inline_expression(op_not_inline, cst.Name("expr")) is False

  # test getattr with no name
  op_getattr: typing.Any = build_mock_op("sw.getattr", results=["%getattr_res"])
  gen.usage_counts["%getattr_res"] = 1
  stmt: typing.Any = gen._wrap_as_statement(op_getattr, cst.Name("attr"))
  assert isinstance(stmt, cst.SimpleStatementLine)

  # test void call [304, 305]
  op_void: typing.Any = build_mock_op("sw.call", results=["%void_res"])
  gen.usage_counts["%void_res"] = 1
  expr: typing.Any = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__")))
  stmt_void: typing.Any = gen._wrap_as_statement(op_void, expr)
  assert isinstance(stmt_void, cst.SimpleStatementLine)
  assert isinstance(stmt_void.body[0], cst.Expr)


def test_generator_is_void_call_branches() -> None:
  """Verify generator is void call branches behavior."""
  gen: typing.Any = MlirToPythonGenerator()
  # [344, 349] - func is Attribute but attr is not __init__
  expr1: typing.Any = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("other")))
  assert not gen._is_void_call(expr1)

  # [346, 349] - attr is __init__ but receiver is not a Call
  expr2: typing.Any = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("__init__")))
  assert not gen._is_void_call(expr2)

  # [347, 349] - receiver is Call but func is not super
  expr3: typing.Any = cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("other")), attr=cst.Name("__init__")))
  assert not gen._is_void_call(expr3)
