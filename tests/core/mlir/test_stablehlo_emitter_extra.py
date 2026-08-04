"""Test module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext


def setup_emitter():
  """Test function."""
  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="torch", target_framework="stablehlo")
  ctx = RewriterContext(semantics=semantics, config=config)  # noqa: F841
  emitter = StableHloEmitter(semantics)
  return emitter, semantics


def test_stablehlo_dummy_import():
  """Test function."""
  emitter, _ = setup_emitter()
  op = emitter._emit_import(cst.Import(names=[cst.ImportAlias(name=cst.Name("math"))]))
  assert op.name == "stablehlo.dummy_import"


def test_resolve_sw_constant_quotes():
  """Test function."""
  emitter, _ = setup_emitter()
  op = OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value='"1.5"')])
  emitter._resolve_sw_constant(op)
  assert op.name == "stablehlo.constant"
  assert op.attributes[0].value == "dense<1.5>"
  assert op.result_types[0].body == "tensor<f32>"


def test_resolve_sw_op_quotes():
  """Test function."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.fake"}}}))
  op = OperationNode(name="sw.op", attributes=[AttributeNode(name="type", value='"torch.fake"')])
  emitter._resolve_sw_op(op)
  assert op.name == "stablehlo.fake"


def test_lookup_stablehlo_op_not_found():
  """Test function."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=None)
  assert emitter._lookup_stablehlo_op("torch.fake") is None


def test_lookup_stablehlo_op_no_variant():
  """Test function."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {}}))
  assert emitter._lookup_stablehlo_op("torch.fake") is None


def test_extract_literal():
  """Test function."""
  emitter, _ = setup_emitter()
  assert emitter._extract_literal(cst.Integer("5")) == 5
  assert emitter._extract_literal(cst.List(elements=[cst.Element(value=cst.Integer("1"))])) == [1]
  assert emitter._extract_literal(cst.Tuple(elements=[cst.Element(value=cst.Integer("2"))])) == [2]
  assert emitter._extract_literal(cst.Pass()) == "%error"


def test_emit_call_lambda():
  """Test function."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}}))
  expr = cst.Call(
    func=cst.Name("reduce"),
    args=[
      cst.Arg(
        value=cst.Lambda(
          params=cst.Parameters(params=[cst.Param(name=cst.Name("a")), cst.Param(name=cst.Name("b"))]), body=cst.Name("a")
        )
      )
    ],
  )
  val, ops = emitter._emit_call(expr)
  assert ops[0].name == "stablehlo.reduce"
  assert len(ops[0].regions) == 1
