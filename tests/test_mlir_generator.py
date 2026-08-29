"""Test module."""

from typing import List, Optional
from unittest import mock

import libcst as cst

from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, ModuleNode, OperationNode, RegionNode, Trivia, ValueNode
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def test_generator_init() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  assert gen.ctx is not None
  assert gen.usage_counts is not None
  assert gen.usage_consumers is not None
  assert gen.deferred_exprs is not None


def test_generate_module() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  # Empty module
  mod: ModuleNode = ModuleNode(body=BlockNode(label="", operations=[]))
  cst_mod: cst.Module = gen.generate(mod)
  assert isinstance(cst_mod, cst.Module)
  assert len(cst_mod.body) == 0


def test_convert_trivia() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  # Test trivia conversion
  trivia_comment: Trivia = Trivia(text="// a comment")
  trivia_percent: Trivia = Trivia(text="% another comment")

  lines: List[cst.EmptyLine] = gen._convert_trivia([trivia_comment, trivia_percent])
  assert len(lines) == 2
  assert getattr(lines[0].comment, "value", None) == "# a comment"
  assert getattr(lines[1].comment, "value", None) == "#% another comment"


def test_scan_block_usage() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  # Op with operands
  op1: OperationNode = OperationNode(name='"sw.op"', operands=[ValueNode(name="%0")])
  op2: OperationNode = OperationNode(name='"sw.op"', operands=[ValueNode(name="%1"), ValueNode(name="%0")])

  mod: ModuleNode = ModuleNode(body=BlockNode(label="", operations=[op1, op2]))
  gen._analyze_module_usage(mod)

  assert gen.usage_counts["%0"] == 2
  assert gen.usage_counts["%1"] == 1
  assert gen.usage_consumers["%0"] == op2
  assert gen.usage_consumers["%1"] == op2


def test_should_inline_expression() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  # no result op
  op_no_res: OperationNode = OperationNode(name='"sw.op"')
  assert not gen._should_inline_expression(op_no_res, cst.Name("x"))

  # sw.constant used
  op_const: OperationNode = OperationNode(name='"sw.constant"', results=[ValueNode(name="%0")])
  gen.usage_counts["%0"] = 1
  assert gen._should_inline_expression(op_const, cst.Name("cst"))

  # sw.getattr used
  op_getattr: OperationNode = OperationNode(name='"sw.getattr"', results=[ValueNode(name="%1")])
  gen.usage_counts["%1"] = 1
  assert gen._should_inline_expression(op_getattr, cst.Name("attr"))

  # void / super
  op_super: OperationNode = OperationNode(
    name='"sw.op"', results=[ValueNode(name="%2")], attributes=[AttributeNode(name="type", value='"super"')]
  )
  assert gen._should_inline_expression(op_super, cst.Name("sup"))

  # statement fusion (sw.return)
  op_normal: OperationNode = OperationNode(name='"sw.op"', results=[ValueNode(name="%3")])
  gen.usage_counts["%3"] = 1
  consumer: OperationNode = OperationNode(name='"sw.return"')
  gen.usage_consumers["%3"] = consumer
  assert gen._should_inline_expression(op_normal, cst.Name("norm"))


def test_resolve_operand() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  # Deferred
  gen.deferred_exprs["%0"] = cst.Name("deferred_var")
  assert getattr(gen._resolve_operand("%0"), "value", None) == "deferred_var"

  # Normal lookup
  gen.ctx.register("%1", hint="foo")
  assert getattr(gen._resolve_operand("%1"), "value", None) == "_foo"

  # Dotted lookup
  def mock_lookup(x: str) -> Optional[str]:
    return "self.layer" if x == "%2" else "_foo"

  with mock.patch.object(gen.ctx, "lookup", side_effect=mock_lookup):
    resolved: cst.BaseExpression = gen._resolve_operand("%2")
    assert isinstance(resolved, cst.Attribute)
    assert getattr(resolved.attr, "value", None) == "layer"


def test_create_expression_from_op() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  op_unknown: OperationNode = OperationNode(name='"unknown"')
  assert gen._create_expression_from_op(op_unknown) is None

  # We mock _expr_sw_constant
  op_const: OperationNode = OperationNode(name='"sw.constant"', attributes=[AttributeNode(name="value", value="42")])
  expr: Optional[cst.BaseExpression] = gen._create_expression_from_op(op_const)
  assert isinstance(expr, cst.Integer)
  assert expr.value == "42"


def test_convert_statement_op() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  op_unknown: OperationNode = OperationNode(name='"unknown"')
  assert gen._convert_statement_op(op_unknown) is None

  op_return: OperationNode = OperationNode(name='"sw.return"')
  stmt: Optional[cst.BaseStatement] = gen._convert_statement_op(op_return)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Return)


def test_wrap_as_statement() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()

  expr: cst.Name = cst.Name("val")

  # No results
  op_no_res: OperationNode = OperationNode(name='"sw.op"')
  stmt: cst.BaseStatement = gen._wrap_as_statement(op_no_res, expr)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)

  # Usage count 0 -> Expr
  op_res: OperationNode = OperationNode(name='"sw.op"', results=[ValueNode(name="%0")])
  gen.usage_counts["%0"] = 0
  stmt = gen._wrap_as_statement(op_res, cst.Name("x"))
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)

  # Void pattern
  void_call: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Call(func=cst.Name("super"), args=[]), attr=cst.Name("__init__")), args=[]
  )
  stmt = gen._wrap_as_statement(op_res, void_call)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)

  # Assignment with hint from 'type'
  op_type_hint: OperationNode = OperationNode(
    name='"sw.op"', results=[ValueNode(name="%1")], attributes=[AttributeNode(name="type", value='"torch.flatten"')]
  )
  gen.usage_counts["%1"] = 1
  stmt = gen._wrap_as_statement(op_type_hint, cst.Name("x"))
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Assign)
  assert getattr(stmt.body[0].targets[0].target, "value", None) == "_flatten"

  # Assignment with hint from 'name'
  op_name_hint: OperationNode = OperationNode(
    name='"sw.getattr"', results=[ValueNode(name="%2")], attributes=[AttributeNode(name="name", value='"my_attr"')]
  )
  gen.usage_counts["%2"] = 1
  stmt = gen._wrap_as_statement(op_name_hint, cst.Name("x"))
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Assign)
  assert getattr(stmt.body[0].targets[0].target, "value", None) == "_my_attr"  # NamingContext prepends _

  # Assignment for sw.constant
  op_const_hint: OperationNode = OperationNode(name='"sw.constant"', results=[ValueNode(name="%3")])
  gen.usage_counts["%3"] = 1
  stmt = gen._wrap_as_statement(op_const_hint, cst.Name("x"))
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Assign)
  assert getattr(stmt.body[0].targets[0].target, "value", None) == "_cst"

  # No results
  op_no_res2: OperationNode = OperationNode(name='"sw.op"')
  stmt_no_res: cst.BaseStatement = gen._wrap_as_statement(op_no_res2, expr)
  assert isinstance(stmt_no_res, cst.SimpleStatementLine)
  assert isinstance(stmt_no_res.body[0], cst.Expr)


def test_convert_block() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  op_const: OperationNode = OperationNode(
    name='"sw.constant"', results=[ValueNode(name="%0")], attributes=[AttributeNode(name="value", value="42")]
  )
  op_return: OperationNode = OperationNode(name='"sw.return"', operands=[ValueNode(name="%0")])

  # Mocking behavior for inline vs statement
  gen.usage_counts["%0"] = 1
  gen.usage_consumers["%0"] = op_return

  block: BlockNode = BlockNode(label="", operations=[op_const, op_return])
  stmts: List[cst.BaseStatement] = gen._convert_block(block)
  assert len(stmts) == 1
  assert isinstance(stmts[0], cst.SimpleStatementLine)
  assert isinstance(stmts[0].body[0], cst.Return)


def test_convert_block_deferred() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  # If inlined but not consumed in a statement immediately
  op_const: OperationNode = OperationNode(
    name='"sw.constant"', results=[ValueNode(name="%0")], attributes=[AttributeNode(name="value", value="42")]
  )
  gen.usage_counts["%0"] = 1
  block: BlockNode = BlockNode(label="", operations=[op_const])
  stmts: List[cst.BaseStatement] = gen._convert_block(block)
  assert len(stmts) == 0
  assert "%0" in gen.deferred_exprs


def test_scan_block_usage_regions() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  b: BlockNode = BlockNode(label="", operations=[OperationNode(name='"sw.op"', operands=[ValueNode(name="%0")])])
  op: OperationNode = OperationNode(name='"sw.parent"', regions=[RegionNode(blocks=[b])])
  gen._analyze_module_usage(ModuleNode(body=BlockNode(label="", operations=[op])))
  assert gen.usage_counts["%0"] == 1


def test_convert_trivia_other() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  lines: List[cst.EmptyLine] = gen._convert_trivia([Trivia(text="other"), Trivia(text="// comment")])
  assert len(lines) == 1
  assert getattr(lines[0].comment, "value", None) == "# comment"


def test_convert_block_stmt_leading() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  # To test stmt_node with leading trivia and no expression
  # We mock _convert_statement_op
  op1: OperationNode = OperationNode(name='"sw.unknown"', leading_trivia=[Trivia(text="// a")])
  op2: OperationNode = OperationNode(name='"sw.statement"', leading_trivia=[Trivia(text="// b")])

  with mock.patch.object(gen, "_create_expression_from_op", return_value=None):
    with mock.patch.object(
      gen,
      "_convert_statement_op",
      side_effect=lambda o: cst.SimpleStatementLine(body=[cst.Pass()]) if o.name == '"sw.statement"' else None,
    ):
      stmts: List[cst.BaseStatement] = gen._convert_block(BlockNode(label="", operations=[op1, op2]))
      assert len(stmts) == 1
      assert len(stmts[0].leading_lines) == 1
      assert getattr(stmts[0].leading_lines[0].comment, "value", None) == "# b"


def test_should_inline_expression_not_statement() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  op_normal: OperationNode = OperationNode(name='"sw.op"', results=[ValueNode(name="%3")])
  gen.usage_counts["%3"] = 1
  consumer: OperationNode = OperationNode(name='"sw.other_consumer"')
  gen.usage_consumers["%3"] = consumer
  assert not gen._should_inline_expression(op_normal, cst.Name("norm"))


def test_create_expression_from_op_branches() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  with mock.patch.object(gen, "_expr_sw_op", return_value="sw.op"):
    assert gen._create_expression_from_op(OperationNode(name='"sw.op"')) == "sw.op"
  with mock.patch.object(gen, "_expr_sw_call", return_value="sw.call"):
    assert gen._create_expression_from_op(OperationNode(name='"sw.call"')) == "sw.call"
  with mock.patch.object(gen, "_expr_sw_getattr", return_value="sw.getattr"):
    assert gen._create_expression_from_op(OperationNode(name='"sw.getattr"')) == "sw.getattr"


def test_convert_statement_op_branches() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  with mock.patch.object(gen, "_convert_class_def", return_value="sw.module"):
    assert gen._convert_statement_op(OperationNode(name='"sw.module"')) == "sw.module"
  with mock.patch.object(gen, "_convert_func_def", return_value="sw.func"):
    assert gen._convert_statement_op(OperationNode(name='"sw.func"')) == "sw.func"
  with mock.patch.object(gen, "_convert_setattr", return_value="sw.setattr"):
    assert gen._convert_statement_op(OperationNode(name='"sw.setattr"')) == "sw.setattr"
  with mock.patch.object(gen, "_convert_import", return_value="sw.import"):
    assert gen._convert_statement_op(OperationNode(name='"sw.import"')) == "sw.import"


def test_wrap_as_statement_branches() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  # sw.getattr without raw_n
  op_getattr_no_n: OperationNode = OperationNode(name='"sw.getattr"', results=[ValueNode(name="%2")])
  gen.usage_counts["%2"] = 1
  stmt: cst.BaseStatement = gen._wrap_as_statement(op_getattr_no_n, cst.Name("x"))
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Assign)
  assert getattr(stmt.body[0].targets[0].target, "value", None) == "_2"  # or whatever auto-assigned


def test_is_void_call_super() -> None:
  """Docstring."""
  gen: MlirToPythonGenerator = MlirToPythonGenerator()
  void_call: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Call(func=cst.Name("super"), args=[]), attr=cst.Name("__init__")), args=[]
  )
  assert gen._is_void_call(void_call)

  op_res: OperationNode = OperationNode(name='"sw.op"', results=[ValueNode(name="%9")])
  gen.usage_counts["%9"] = 1
  stmt: cst.BaseStatement = gen._wrap_as_statement(op_res, void_call)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)
