"""Test module."""

import libcst as cst
from ml_switcheroo.core.mlir.emitter_expr import MlirEmitterExprMixin
from ml_switcheroo.core.mlir.cst import ValueNode, OperationNode
from typing import Dict, List, Optional


class DummyCtx:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.count: int = 0
    self.vars: Dict[str, ValueNode] = {"foo": ValueNode(name="%foo")}

  def lookup(self, name: str) -> Optional[ValueNode]:
    """Test element."""
    return self.vars.get(name)

  def allocate_ssa(self, prefix: str = "%") -> ValueNode:
    """Test element."""
    self.count += 1
    return ValueNode(name=f"{prefix}{self.count}")


class DummyEmitter(MlirEmitterExprMixin):
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.ctx: DummyCtx = DummyCtx()

  def _flatten_attr(self, attr: cst.BaseExpression) -> Optional[str]:
    if isinstance(attr, cst.Name):
      return attr.value
    elif isinstance(attr, cst.Attribute):
      base: Optional[str] = self._flatten_attr(attr.value)
      if base:
        return f"{base}.{attr.attr.value}"
    return None

  def _get_binop_str(self, op: cst.BaseBinaryOp) -> str:
    if isinstance(op, cst.Add):
      return "add"
    if isinstance(op, cst.Multiply):
      return "mul"
    return "unknown"


def test_emit_name() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  name_node: cst.Name = cst.Name(value="foo")
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(name_node)
  assert val.name == "%foo"
  assert len(ops) == 0

  name_node2: cst.Name = cst.Name(value="bar")
  val2: ValueNode
  ops2: List[OperationNode]
  val2, ops2 = emitter._emit_expression(name_node2)
  assert val2.name == "@bar"


def test_emit_call() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  # static op call: tf.add(x, y=z)
  call_node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("add")),
    args=[cst.Arg(value=cst.Name("foo")), cst.Arg(keyword=cst.Name("y"), value=cst.Name("foo"))],
  )
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  assert val.name == "%1"
  assert len(ops) == 1
  assert ops[0].name == "sw.op"


def test_emit_method_call() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  # object method call: foo.add() where foo is a local variable
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("add")), args=[])
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  # ops should have sw.getattr and sw.call
  assert len(ops) == 2
  assert ops[0].name == "sw.getattr"
  assert ops[1].name == "sw.call"


def test_emit_func_call() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  # direct func call: foo() where foo is local
  call_node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.call"


def test_emit_binary_op() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  bin_node: cst.BinaryOperation = cst.BinaryOperation(left=cst.Name("foo"), operator=cst.Add(), right=cst.Name("foo"))
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(bin_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.op"
  attrs = ops[0].attributes
  assert any(a.name == "type" and a.value == '"binop.add"' for a in attrs)


def test_emit_constants() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  int_node: cst.Integer = cst.Integer("42")
  val1: ValueNode
  ops1: List[OperationNode]
  val1, ops1 = emitter._emit_expression(int_node)
  assert len(ops1) == 1

  float_node: cst.Float = cst.Float("3.14")
  val2: ValueNode
  ops2: List[OperationNode]
  val2, ops2 = emitter._emit_expression(float_node)
  assert len(ops2) == 1


def test_emit_error() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()

  class DummyNode(cst.BaseExpression):
    def _visit_and_replace_children(self, visitor: object) -> cst.CSTNode:
      return self

    def _codegen_impl(self, state: object, default_semi: bool = False) -> None:
      pass

  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(DummyNode())
  assert val.name == "%error"


def test_annotation_to_string() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  n: cst.Name = cst.Name("int")
  assert emitter._annotation_to_string(n) == "int"

  attr: cst.Attribute = cst.Attribute(value=cst.Name("typing"), attr=cst.Name("Any"))
  assert emitter._annotation_to_string(attr) == "typing.Any"

  class DummyNode(cst.CSTNode):
    def _visit_and_replace_children(self, visitor: object) -> cst.CSTNode:
      return self

    def _codegen_impl(self, state: object, default_semi: bool = False) -> None:
      pass

  assert emitter._annotation_to_string(DummyNode()) == "Any"


def test_emit_call_other() -> None:
  """Test element."""
  emitter: DummyEmitter = DummyEmitter()
  # call of call: foo()()
  call_node: cst.Call = cst.Call(func=cst.Call(func=cst.Name("foo"), args=[]), args=[])
  val: ValueNode
  ops: List[OperationNode]
  val, ops = emitter._emit_expression(call_node)
  # wait, _emit_expression will fall through to return ValueNode(name="%error"), []
  assert val.name == "%error"
