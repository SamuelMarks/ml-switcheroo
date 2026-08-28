"""Tests for StableHloEmitter edge cases and statement resolution."""

import libcst as cst
import typing
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.cst import OperationNode, TypeNode
from unittest.mock import MagicMock


def test_while_return() -> None:
  """Test emitting while statement containing return."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "while True:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_while(typing.cast(cst.While, tree.body[0]))


def test_if_true_return() -> None:
  """Test emitting if statement with a return in the main branch."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_if_else_return() -> None:
  """Test emitting if statement with a return in the else branch."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  pass\nelse:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_if_elif_return() -> None:
  """Test emitting if statement with a return in an elif branch."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  pass\nelif False:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_sw_constant_existing_type() -> None:
  """Test resolving sw.constant when it already has a type."""
  emitter = StableHloEmitter(MagicMock())
  op = OperationNode(name="sw.constant", attributes=[], result_types=[TypeNode("tensor<f32>")])
  import ml_switcheroo.core.mlir.cst as m_cst

  op.attributes.append(m_cst.AttributeNode("value", "5.0"))
  emitter._resolve_sw_constant(op)


def test_sw_op_existing_type() -> None:
  """Test resolving sw.op when it already has a type."""
  mgr = MagicMock()
  mgr.get_definition.return_value = ("torch", {"variants": {"jax": {"api": "jax.numpy.abs"}}})
  emitter = StableHloEmitter(mgr)
  op = OperationNode(name="sw.op", attributes=[], result_types=[TypeNode("tensor<f32>")])
  import ml_switcheroo.core.mlir.cst as m_cst

  op.attributes.append(m_cst.AttributeNode("type", '"torch.abs"'))
  emitter._lookup_stablehlo_op = MagicMock(return_value="stablehlo.abs")  # type: ignore
  emitter._resolve_sw_op(op)


def test_call_non_name() -> None:
  """Test emitting call for non-name node."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "(lambda x: x)()"
  tree: cst.Module = cst.parse_module(code)
  expr: typing.Any = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, tree.body[0]).body[0]).value
  emitter._emit_call(expr)


def test_func_def_multiple_params() -> None:
  """Test emitting function definition with multiple parameters."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "def foo(a, b):\n  pass"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_func_def(typing.cast(cst.FunctionDef, tree.body[0]))


def test_stablehlo_parser() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser

  parser = StableHloParser("module {}")
  mod: typing.Any = parser.parse()
  assert len(mod.body.operations) == 1
  assert mod.body.operations[0].name == "module"
