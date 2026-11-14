"""Consolidated unit test suite for ``MlirGenHelpers`` and ``MlirToPythonGenerator``."""

from unittest import mock

import libcst as cst

from ml_switcheroo.core.cst.base import Trivia as BaseTrivia
from ml_switcheroo.core.mlir.cst import (
  AttributeNode,
  BlockNode,
  ModuleNode,
  OperationNode,
  RegionNode,
  Trivia,
  TypeNode,
  ValueNode,
)
from ml_switcheroo.core.mlir.gen_base import MlirGenHelpers
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator


def gen_code(node: ModuleNode) -> str:
  """Generate formatted Python source code from an MLIR ``ModuleNode``.

  Args:
      node: MLIR ``ModuleNode`` to lower to Python.

  Returns:
      Generated Python source code string.
  """
  gen = MlirToPythonGenerator()
  return gen.generate(node).code


def test_mlir_gen_helpers_get_attr_and_dotted_name() -> None:
  """Verify ``MlirGenHelpers.get_attr`` and ``create_dotted_name`` across scalar, list, and empty inputs."""
  helpers = MlirGenHelpers()
  op = OperationNode(name="test.op")
  assert helpers.get_attr(op, "missing") is None

  op2 = OperationNode(
    name="test.op",
    attributes=[AttributeNode(name="other", value="val"), AttributeNode(name="foo", value="bar")],
  )
  assert helpers.get_attr(op2, "foo") == "bar"

  op3 = OperationNode(name="test.op", attributes=[AttributeNode(name="arr", value=["1", "2"])])
  assert helpers.get_attr(op3, "arr") == "[1, 2]"

  op4 = OperationNode(
    name="test",
    attributes=[AttributeNode(name="k", value="[v1, v2]", type_annotation="str")],
  )
  assert helpers.get_attr(op4, "k") == "[v1, v2]"

  empty_node = helpers.create_dotted_name("")
  assert isinstance(empty_node, cst.Name)
  assert empty_node.value == "unknown"

  single_node = helpers.create_dotted_name("torch")
  assert isinstance(single_node, cst.Name)
  assert single_node.value == "torch"

  dotted_node = helpers.create_dotted_name("torch.nn.functional")
  assert isinstance(dotted_node, cst.Attribute)
  assert dotted_node.attr.value == "functional"
  assert isinstance(dotted_node.value, cst.Attribute)
  assert dotted_node.value.attr.value == "nn"
  assert isinstance(dotted_node.value.value, cst.Name)
  assert dotted_node.value.value.value == "torch"


def test_generator_init_and_empty_module() -> None:
  """Verify ``MlirToPythonGenerator`` initializes composed delegates and handles empty modules."""
  gen = MlirToPythonGenerator()
  assert gen.ctx is not None
  assert gen.helpers is not None
  assert gen.expr_generator is not None
  assert gen.stmt_generator is not None

  mod = ModuleNode(body=BlockNode(label="", operations=[]))
  cst_mod = gen.generate(mod)
  assert isinstance(cst_mod, cst.Module)
  assert len(cst_mod.body) == 0


def test_module_func_and_expression_end_to_end() -> None:
  """Verify class, function, constant, getattr, sw.op, and sw.call lowering end-to-end."""
  cls_op = OperationNode(name="sw.module", attributes=[AttributeNode(name="sym_name", value='"MyClass"')])
  cls_code = gen_code(ModuleNode(body=BlockNode(label="", operations=[cls_op])))
  assert "class MyClass:" in cls_code
  assert "pass" in cls_code

  ret_op = OperationNode(name="sw.return", operands=[ValueNode(name="%x")])
  body_blk = BlockNode(label="^entry", arguments=[(ValueNode(name="%x"), TypeNode(body="!sw.unk"))], operations=[ret_op])
  func_op = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="sym_name", value='"forward"')],
    regions=[RegionNode(blocks=[body_blk])],
  )
  func_code = gen_code(ModuleNode(body=BlockNode(label="", operations=[func_op])))
  assert "def forward(x):" in func_code
  assert "return x" in func_code

  add_op = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    operands=[ValueNode(name="%a"), ValueNode(name="%b")],
    attributes=[AttributeNode(name="type", value='"torch.add"')],
  )
  use_op = OperationNode(name="sw.return", operands=[ValueNode(name="%0")])
  assert "return torch.add(_a, _b)" in gen_code(ModuleNode(body=BlockNode("", operations=[add_op, use_op])))

  cst_op = OperationNode(
    name="sw.constant",
    results=[ValueNode(name="%c")],
    attributes=[AttributeNode(name="value", value="1")],
  )
  assert "return 1" in gen_code(
    ModuleNode(body=BlockNode("", operations=[cst_op, OperationNode(name="sw.return", operands=[ValueNode(name="%c")])]))
  )

  attr_op = OperationNode(
    name="sw.getattr",
    results=[ValueNode(name="%attr")],
    operands=[ValueNode(name="%self")],
    attributes=[AttributeNode(name="name", value='"layer"')],
  )
  assert "return _self.layer" in gen_code(
    ModuleNode(
      body=BlockNode("", operations=[attr_op, OperationNode(name="sw.return", operands=[ValueNode(name="%attr")])])
    )
  )

  call_op = OperationNode(
    name="sw.call",
    results=[ValueNode(name="%res")],
    operands=[ValueNode(name="%func"), ValueNode(name="%arg")],
  )
  assert "return _func(_arg)" in gen_code(
    ModuleNode(
      body=BlockNode("", operations=[call_op, OperationNode(name="sw.return", operands=[ValueNode(name="%res")])])
    )
  )


def test_trivia_conversion_and_restoration() -> None:
  """Verify ``//`` and ``%`` trivia comments convert cleanly to Python ``#`` comments."""
  gen = MlirToPythonGenerator()
  lines = gen._convert_trivia([Trivia(text="other"), Trivia(text="// a comment"), Trivia(text="% another comment")])
  assert len(lines) == 2
  assert lines[0].comment is not None
  assert lines[0].comment.value == "# a comment"
  assert lines[1].comment is not None
  assert lines[1].comment.value == "#% another comment"

  op_base = OperationNode(name="sw.return", leading_trivia=[BaseTrivia("// My Comment")])
  assert "# My Comment" in gen_code(ModuleNode(body=BlockNode("", operations=[op_base])))

  op_pct = OperationNode(name="sw.return", leading_trivia=[Trivia(text="% comment")])
  assert "#% comment" in gen_code(ModuleNode(body=BlockNode("", operations=[op_pct])))


def test_scan_usage_inline_and_operand_resolution() -> None:
  """Verify SSA usage scanning, inline expression folding, and dotted operand resolution."""
  gen = MlirToPythonGenerator()
  op1 = OperationNode(name='"sw.op"', operands=[ValueNode(name="%0")])
  op2 = OperationNode(name='"sw.op"', operands=[ValueNode(name="%1"), ValueNode(name="%0")])
  nested_blk = BlockNode(label="", operations=[OperationNode(name='"sw.op"', operands=[ValueNode(name="%0")])])
  parent_op = OperationNode(name='"sw.parent"', regions=[RegionNode(blocks=[nested_blk])])
  gen._analyze_module_usage(ModuleNode(body=BlockNode(label="", operations=[op1, op2, parent_op])))

  assert gen.usage_counts["%0"] == 3
  assert gen.usage_counts["%1"] == 1
  assert gen.usage_consumers["%0"] == nested_blk.operations[0]
  assert gen.usage_consumers["%1"] == op2

  assert not gen._should_inline_expression(OperationNode(name='"sw.op"'), cst.Name("x"))

  op_const = OperationNode(name='"sw.constant"', results=[ValueNode(name="%c0")])
  gen.usage_counts["%c0"] = 1
  assert gen._should_inline_expression(op_const, cst.Name("cst"))

  op_getattr = OperationNode(name='"sw.getattr"', results=[ValueNode(name="%g0")])
  gen.usage_counts["%g0"] = 1
  assert gen._should_inline_expression(op_getattr, cst.Name("attr"))

  op_super = OperationNode(
    name='"sw.op"',
    results=[ValueNode(name="%s0")],
    attributes=[AttributeNode(name="type", value='"super"')],
  )
  assert gen._should_inline_expression(op_super, cst.Name("sup"))

  op_ret_consumer = OperationNode(name='"sw.op"', results=[ValueNode(name="%r0")])
  gen.usage_counts["%r0"] = 1
  gen.usage_consumers["%r0"] = OperationNode(name='"sw.return"')
  assert gen._should_inline_expression(op_ret_consumer, cst.Name("norm"))

  op_other_consumer = OperationNode(name='"sw.op"', results=[ValueNode(name="%o0")])
  gen.usage_counts["%o0"] = 1
  gen.usage_consumers["%o0"] = OperationNode(name='"sw.other"')
  assert not gen._should_inline_expression(op_other_consumer, cst.Name("norm"))

  gen.deferred_exprs["%d0"] = cst.Name("deferred_var")
  resolved_def = gen._resolve_operand("%d0")
  assert isinstance(resolved_def, cst.Name)
  assert resolved_def.value == "deferred_var"

  gen.ctx.register("%n1", hint="foo")
  resolved_n1 = gen._resolve_operand("%n1")
  assert isinstance(resolved_n1, cst.Name)
  assert resolved_n1.value == "_foo"

  with mock.patch.object(gen.ctx, "lookup", return_value="self.layer"):
    resolved_dot = gen._resolve_operand("%n2")
    assert isinstance(resolved_dot, cst.Attribute)
    assert resolved_dot.attr.value == "layer"


def test_wrap_as_statement_and_delegate_routing() -> None:
  """Verify statement wrapping, void ``super().__init__`` detection, and delegate dispatch."""
  gen = MlirToPythonGenerator()

  stmt_no_res = gen._wrap_as_statement(OperationNode(name='"sw.op"'), cst.Name("val"))
  assert isinstance(stmt_no_res.body[0], cst.Expr)

  op_zero_use = OperationNode(name='"sw.op"', results=[ValueNode(name="%0")])
  gen.usage_counts["%0"] = 0
  assert isinstance(gen._wrap_as_statement(op_zero_use, cst.Name("x")).body[0], cst.Expr)

  void_call = cst.Call(
    func=cst.Attribute(value=cst.Call(func=cst.Name("super"), args=[]), attr=cst.Name("__init__")),
    args=[],
  )
  assert gen._is_void_call(void_call)
  gen.usage_counts["%0"] = 1
  assert isinstance(gen._wrap_as_statement(op_zero_use, void_call).body[0], cst.Expr)

  op_type_hint = OperationNode(
    name='"sw.op"',
    results=[ValueNode(name="%1")],
    attributes=[AttributeNode(name="type", value='"torch.flatten"')],
  )
  gen.usage_counts["%1"] = 1
  stmt_t = gen._wrap_as_statement(op_type_hint, cst.Name("x"))
  assign_t = stmt_t.body[0]
  assert isinstance(assign_t, cst.Assign)
  assert isinstance(assign_t.targets[0].target, cst.Name)
  assert assign_t.targets[0].target.value == "_flatten"

  op_name_hint = OperationNode(
    name='"sw.getattr"',
    results=[ValueNode(name="%2")],
    attributes=[AttributeNode(name="name", value='"my_attr"')],
  )
  gen.usage_counts["%2"] = 1
  stmt_n = gen._wrap_as_statement(op_name_hint, cst.Name("x"))
  assign_n = stmt_n.body[0]
  assert isinstance(assign_n, cst.Assign)
  assert isinstance(assign_n.targets[0].target, cst.Name)
  assert assign_n.targets[0].target.value == "_my_attr"

  op_no_name_hint = OperationNode(name='"sw.getattr"', results=[ValueNode(name="%2b")])
  gen.usage_counts["%2b"] = 1
  stmt_nn = gen._wrap_as_statement(op_no_name_hint, cst.Name("x"))
  assert isinstance(stmt_nn.body[0], cst.Assign)

  op_const_hint = OperationNode(name='"sw.constant"', results=[ValueNode(name="%3")])
  gen.usage_counts["%3"] = 1
  stmt_c = gen._wrap_as_statement(op_const_hint, cst.Name("x"))
  assign_c = stmt_c.body[0]
  assert isinstance(assign_c, cst.Assign)
  assert isinstance(assign_c.targets[0].target, cst.Name)
  assert assign_c.targets[0].target.value == "_cst"

  # Deferred trivia propagation & statement leading trivia
  op_def = OperationNode(
    name='"sw.constant"',
    results=[ValueNode(name="%d")],
    attributes=[AttributeNode(name="value", value="42")],
    leading_trivia=[Trivia(text="// deferred comment")],
  )
  op_ret = OperationNode(name='"sw.return"', operands=[ValueNode(name="%d")])
  gen.usage_counts["%d"] = 1
  gen.usage_consumers["%d"] = op_ret
  stmts = gen._convert_block(BlockNode(label="", operations=[op_def, op_ret]))
  assert len(stmts) == 1
  assert len(stmts[0].leading_lines) == 1

  op_stmt_triv = OperationNode(name='"sw.return"', leading_trivia=[Trivia(text="// b")])
  stmts_triv = gen._convert_block(BlockNode(label="", operations=[OperationNode(name='"sw.unknown"'), op_stmt_triv]))
  assert len(stmts_triv) == 1
  assert len(stmts_triv[0].leading_lines) == 1

  assert gen._create_expression_from_op(OperationNode(name='"unknown"')) is None
  assert gen._convert_statement_op(OperationNode(name='"unknown"')) is None
  assert isinstance(
    gen._convert_statement_op(
      OperationNode(
        name='"sw.import"',
        attributes=[AttributeNode(name="names", value='["os"]'), AttributeNode(name="aliases", value='[""]')],
      )
    ),
    cst.SimpleStatementLine,
  )
  assert isinstance(
    gen._convert_statement_op(OperationNode(name='"sw.setattr"')),
    cst.SimpleStatementLine,
  )

  class _TriviaWithContent:
    """Trivia stub exposing a `content` attribute."""

    def __init__(self, content: str) -> None:
      """Initialize `content` string.

      Args:
          content: Comment text stored in `content`.
      """
      self.content = content

  class _TriviaWithText:
    """Trivia stub exposing a `text` attribute."""

    def __init__(self, text: str) -> None:
      """Initialize `text` string.

      Args:
          text: Comment text stored in `text`.
      """
      self.text = text

  assert len(gen._convert_trivia([_TriviaWithContent("// comment from content")])) == 1
  assert len(gen._convert_trivia([_TriviaWithText("// comment from text")])) == 1
  assert gen._is_void_call(cst.ensure_type(cst.parse_expression("self.__init__()"), cst.Call)) is False
  assert gen._is_void_call(cst.ensure_type(cst.parse_expression("other().__init__()"), cst.Call)) is False
