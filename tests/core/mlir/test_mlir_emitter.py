"""Test suite for the Mlir Emitter module."""

import typing
import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter, SSAContext
from ml_switcheroo.core.mlir.cst import ModuleNode, ValueNode


def test_ssa_context() -> None:
  """Verifies the behavior of ssa context."""
  ctx = SSAContext()
  ctx.enter_scope()
  val1: ValueNode = ctx.allocate_ssa()
  assert val1.name == "%0"
  ctx.declare("foo", val1)
  assert ctx.lookup("foo") == val1
  assert ctx.lookup("bar") is None
  ctx.exit_scope()
  assert ctx.lookup("foo") is None
  ctx.exit_scope()
  assert len(ctx._scopes) == 1


def test_emitter_empty_module() -> None:
  """Verifies the behavior of emitter empty module."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 0


def test_emitter_module_trivia() -> None:
  """Verifies the behavior of emitter module trivia."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("# header\n\nx = 1")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 1
  trivia: list[typing.Any] = mod.body.operations[0].leading_trivia
  assert len(trivia) >= 1
  assert any((t.text.startswith("// header") for t in trivia))


def test_emitter_import() -> None:
  """Verifies the behavior of emitter import."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("import os\nfrom math import sqrt as root")
  mod: ModuleNode = emitter.convert(tree)
  ops: list[typing.Any] = mod.body.operations
  assert len(ops) == 2
  assert ops[0].name == "sw.import"
  assert ops[1].name == "sw.import"


def test_emitter_import_star() -> None:
  """Verifies the behavior of emitter import star."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("from os import *")
  mod: ModuleNode = emitter.convert(tree)
  ops: list[typing.Any] = mod.body.operations
  assert len(ops) == 1
  assert ops[0].name == "sw.import"
  assert any((a.name == "names" and a.value == ['"*"'] for a in ops[0].attributes))


def test_emitter_assign() -> None:
  """Verifies the behavior of emitter assign."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("x = 1")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 1
  assert emitter.ctx.lookup("x") is not None


def test_emitter_assign_attr() -> None:
  """Verifies the behavior of emitter assign attribute."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("self.x = 1")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 1


def test_emitter_assign_attr_known() -> None:
  """Verifies the behavior of emitter assign attribute known."""
  emitter = PythonToMlirEmitter()
  val: ValueNode = emitter.ctx.allocate_ssa()
  emitter.ctx.declare("self", val)
  tree: cst.Module = cst.parse_module("self.x = 1")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 2
  assert mod.body.operations[1].name == "sw.setattr"


def test_emitter_return() -> None:
  """Verifies the behavior of emitter return."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("return 42")
  mod: ModuleNode = emitter.convert(tree)
  ops: list[typing.Any] = mod.body.operations
  assert len(ops) == 2
  assert ops[1].name == "sw.return"


def test_emitter_return_empty() -> None:
  """Verifies the behavior of emitter return empty."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("return")
  mod: ModuleNode = emitter.convert(tree)
  assert len(mod.body.operations) == 1
  assert mod.body.operations[0].name == "sw.return"


def test_emitter_expr() -> None:
  """Verifies the behavior of emitter expr."""
  emitter = PythonToMlirEmitter()
  tree: cst.Module = cst.parse_module("1 + 1")
  mod: ModuleNode = emitter.convert(tree)
  ops: list[typing.Any] = mod.body.operations
  assert len(ops) == 3


def test_get_binop_str() -> None:
  """Gets binop string."""
  emitter = PythonToMlirEmitter()
  assert emitter._get_binop_str(cst.Add()) == "add"
  assert emitter._get_binop_str(cst.Subtract()) == "sub"
  assert emitter._get_binop_str(cst.Multiply()) == "mul"
  assert emitter._get_binop_str(cst.Divide()) == "div"
  assert emitter._get_binop_str(cst.FloorDivide()) == "floordiv"
  assert emitter._get_binop_str(cst.Modulo()) == "mod"
  assert emitter._get_binop_str(cst.Power()) == "pow"
  assert emitter._get_binop_str(cst.MatrixMultiply()) == "matmul"
  assert emitter._get_binop_str(cst.LeftShift()) == "lshift"
  assert emitter._get_binop_str(cst.RightShift()) == "rshift"
  assert emitter._get_binop_str(cst.BitAnd()) == "and"
  assert emitter._get_binop_str(cst.BitOr()) == "or"
  assert emitter._get_binop_str(cst.BitXor()) == "xor"

  class MockOp(cst.BaseBinaryOp):
    """Mock."""

    def _codegen_impl(self) -> None:
      """Codegen."""
      pass

    def _visit_and_replace_children(self, visitor: typing.Any) -> typing.Any:
      """Visit."""
      pass

  assert emitter._get_binop_str(MockOp()) == "unknown"
