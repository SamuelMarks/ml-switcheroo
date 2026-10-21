"""Test suite for mlir emitter expr."""

import libcst as cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode
from ml_switcheroo.core.mlir.emitter_expr import MlirEmitterExprMixin


class DummyCtx:
  """Test helper class DummyCtx."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.count: int = 0
    self.vars: dict[str, ValueNode] = {"foo": ValueNode(name="%foo")}

  def lookup(self, name: str) -> ValueNode | None:
    """Execute lookup helper logic.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.vars.get(name)

  def allocate_ssa(self, prefix: str = "%") -> ValueNode:
    """Execute allocate ssa helper logic.

    Args:
        prefix: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    self.count += 1
    return ValueNode(name=f"{prefix}{self.count}")


class DummyEmitter(MlirEmitterExprMixin):
  """Test helper class DummyEmitter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "ctx",
      getattr({"v": getattr({"v": DummyCtx}, "__getitem__")("v")()}, "__getitem__")("v"),
    )

  def _flatten_attr(self, attr: cst.BaseExpression) -> str | None:
    """Execute flatten attr helper logic.

    Returns:
        Computed test result or fixture value.
    """
    if isinstance(attr, cst.Name):
      return attr.value
    elif isinstance(attr, cst.Attribute):
      base: str | None = self._flatten_attr(attr.value)
      if base:
        return f"{base}.{attr.attr.value}"
    return None

  def _get_binop_str(self, op: cst.BaseBinaryOp) -> str:
    """Execute get binop str helper logic.

    Args:
        op: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if isinstance(op, cst.Add):
      return "add"
    if isinstance(op, cst.Multiply):
      return "mul"
    return "unknown"


def test_emit_name() -> None:
  """Verify emit name behavior."""
  emitter: DummyEmitter = DummyEmitter()
  name_node: cst.Name = cst.Name(value="foo")
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(name_node)
  assert val.name == "%foo"
  assert len(ops) == 0

  name_node2: cst.Name = cst.Name(value="bar")
  val2: ValueNode
  ops2: list[OperationNode]
  val2, ops2 = emitter._emit_expression(name_node2)
  assert val2.name == "@bar"


def test_emit_call() -> None:
  """Verify emit call behavior."""
  emitter: DummyEmitter = DummyEmitter()
  # static op call: tf.add(x, y=z)
  call_node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("add")),
    args=[cst.Arg(value=cst.Name("foo")), cst.Arg(keyword=cst.Name("y"), value=cst.Name("foo"))],
  )
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  assert val.name == "%1"
  assert len(ops) == 1
  assert ops[0].name == "sw.op"


def test_emit_method_call() -> None:
  """Verify emit method call behavior."""
  emitter: DummyEmitter = DummyEmitter()
  # object method call: foo.add() where foo is a local variable
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("add")), args=[])
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  # ops should have sw.getattr and sw.call
  assert len(ops) == 2
  assert ops[0].name == "sw.getattr"
  assert ops[1].name == "sw.call"


def test_emit_func_call() -> None:
  """Verify emit func call behavior."""
  emitter: DummyEmitter = DummyEmitter()
  # direct func call: foo() where foo is local
  call_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.call"


def test_emit_binary_op() -> None:
  """Verify emit binary op behavior."""
  emitter: DummyEmitter = DummyEmitter()
  bin_node: cst.BinaryOperation = cst.BinaryOperation(left=cst.Name("foo"), operator=cst.Add(), right=cst.Name("foo"))
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(bin_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.op"
  attrs = ops[0].attributes
  assert any(a.name == "type" and a.value == '"binop.add"' for a in attrs)


def test_emit_constants() -> None:
  """Verify emit constants behavior."""
  emitter: DummyEmitter = DummyEmitter()
  int_node: cst.Integer = cst.Integer("42")
  val1: ValueNode
  ops1: list[OperationNode]
  val1, ops1 = emitter._emit_expression(int_node)
  assert len(ops1) == 1

  float_node: cst.Float = cst.Float("3.14")
  val2: ValueNode
  ops2: list[OperationNode]
  val2, ops2 = emitter._emit_expression(float_node)
  assert len(ops2) == 1


def test_emit_error() -> None:
  """Verify emit error behavior."""
  emitter: DummyEmitter = DummyEmitter()

  class DummyNode(cst.BaseExpression):
    """Test helper class DummyNode."""

    def _visit_and_replace_children(self, visitor: cst.CSTVisitorT) -> cst.CSTNode:
      """Execute visit and replace children helper logic.

      Args:
          visitor: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return self

    def _codegen_impl(self, state: CodegenState, default_semi: bool = False) -> None:
      """Execute codegen impl helper logic.

      Args:
          state: Test argument or fixture.
          default_semi: Test argument or fixture.
      """
      pass

  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(DummyNode())
  assert val.name == "%error"


def test_annotation_to_string() -> None:
  """Verify annotation to string behavior."""
  emitter: DummyEmitter = DummyEmitter()
  n: cst.Name = cst.Name("int")
  assert emitter._annotation_to_string(n) == "int"

  attr: cst.Attribute = cst.Attribute(value=cst.Name("typing"), attr=cst.Name("Any"))
  assert emitter._annotation_to_string(attr) == "typing.Any"

  class DummyNode(cst.CSTNode):
    """Test helper class DummyNode."""

    def _visit_and_replace_children(self, visitor: cst.CSTVisitorT) -> cst.CSTNode:
      """Execute visit and replace children helper logic.

      Args:
          visitor: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return self

    def _codegen_impl(self, state: CodegenState, default_semi: bool = False) -> None:
      """Execute codegen impl helper logic.

      Args:
          state: Test argument or fixture.
          default_semi: Test argument or fixture.
      """
      pass

  assert emitter._annotation_to_string(DummyNode()) == "Any"


def test_emit_call_other() -> None:
  """Verify emit call other behavior."""
  emitter: DummyEmitter = DummyEmitter()
  # call of call: foo()()
  call_node: cst.Call = cst.Call(func=cst.Call(func=cst.Name("foo"), args=[]), args=[])
  val: ValueNode
  ops: list[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  # wait, _emit_expression will fall through to return ValueNode(name="%error"), []
  assert val.name == "%error"
