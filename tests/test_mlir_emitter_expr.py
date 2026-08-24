"""Test module."""

import libcst as cst
from typing import Any
from ml_switcheroo.core.mlir.emitter_expr import MlirEmitterExprMixin
from ml_switcheroo.core.mlir.cst import ValueNode


class DummyCtx:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.count = 0
    self.vars = {"foo": ValueNode(name="%foo")}

  def lookup(self, name: str) -> Any:
    """Test element."""
    return self.vars.get(name)

  def allocate_ssa(self, prefix="%"):
    """Test element."""
    self.count += 1
    return ValueNode(name=f"{prefix}{self.count}")


class DummyEmitter(MlirEmitterExprMixin):
  """Test element."""

  def __init__(self):
    """Test element."""
    self.ctx = DummyCtx()

  def _flatten_attr(self, attr: Any) -> Any:
    if isinstance(attr, cst.Name):
      return attr.value
    elif isinstance(attr, cst.Attribute):
      base = self._flatten_attr(attr.value)
      if base:
        return f"{base}.{attr.attr.value}"
    return None

  def _get_binop_str(self, op: Any) -> str:
    if isinstance(op, cst.Add):
      return "add"
    if isinstance(op, cst.Multiply):
      return "mul"
    return "unknown"


def test_emit_name():
  """Test element."""
  emitter = DummyEmitter()
  name_node = cst.Name(value="foo")
  val, ops = emitter._emit_expression(name_node)
  assert val.name == "%foo"
  assert len(ops) == 0

  name_node2 = cst.Name(value="bar")
  val2, ops2 = emitter._emit_expression(name_node2)
  assert val2.name == "@bar"


def test_emit_call():
  """Test element."""
  emitter = DummyEmitter()
  # static op call: tf.add(x, y=z)
  call_node = cst.Call(
    func=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("add")),
    args=[cst.Arg(value=cst.Name("foo")), cst.Arg(keyword=cst.Name("y"), value=cst.Name("foo"))],
  )
  val, ops = emitter._emit_expression(call_node)
  assert val.name == "%1"
  assert len(ops) == 1
  assert ops[0].name == "sw.op"


def test_emit_method_call():
  """Test element."""
  emitter = DummyEmitter()
  # object method call: foo.add() where foo is a local variable
  call_node = cst.Call(func=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("add")), args=[])
  val, ops = emitter._emit_expression(call_node)
  # ops should have sw.getattr and sw.call
  assert len(ops) == 2
  assert ops[0].name == "sw.getattr"
  assert ops[1].name == "sw.call"


def test_emit_func_call():
  """Test element."""
  emitter = DummyEmitter()
  # direct func call: foo() where foo is local
  call_node = cst.Call(func=cst.Name("foo"), args=[])
  val, ops = emitter._emit_expression(call_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.call"


def test_emit_binary_op():
  """Test element."""
  emitter = DummyEmitter()
  bin_node = cst.BinaryOperation(left=cst.Name("foo"), operator=cst.Add(), right=cst.Name("foo"))
  val, ops = emitter._emit_expression(bin_node)
  assert len(ops) == 1
  assert ops[0].name == "sw.op"
  attrs = ops[0].attributes
  assert any(a.name == "type" and a.value == '"binop.add"' for a in attrs)


def test_emit_constants():
  """Test element."""
  emitter = DummyEmitter()
  int_node = cst.Integer("42")
  val1, ops1 = emitter._emit_expression(int_node)
  assert len(ops1) == 1

  float_node = cst.Float("3.14")
  val2, ops2 = emitter._emit_expression(float_node)
  assert len(ops2) == 1


def test_emit_error():
  """Test element."""
  emitter = DummyEmitter()

  class DummyNode(cst.BaseExpression):
    def _visit_and_replace_children(self, visitor):
      return self

    def _codegen_impl(self, state, default_semi=False):
      pass

  val, ops = emitter._emit_expression(DummyNode())
  assert val.name == "%error"


def test_annotation_to_string():
  """Test element."""
  emitter = DummyEmitter()
  n = cst.Name("int")
  assert emitter._annotation_to_string(n) == "int"

  attr = cst.Attribute(value=cst.Name("typing"), attr=cst.Name("Any"))
  assert emitter._annotation_to_string(attr) == "typing.Any"

  class DummyNode(cst.CSTNode):
    def _visit_and_replace_children(self, visitor):
      return self

    def _codegen_impl(self, state, default_semi=False):
      pass

  assert emitter._annotation_to_string(DummyNode()) == "Any"


def test_emit_call_other():
  """Test element."""
  emitter = DummyEmitter()
  # call of call: foo()()
  call_node = cst.Call(func=cst.Call(func=cst.Name("foo"), args=[]), args=[])
  val, ops = emitter._emit_expression(call_node)
  # wait, _emit_expression will fall through to return ValueNode(name="%error"), []
  assert val.name == "%error"
