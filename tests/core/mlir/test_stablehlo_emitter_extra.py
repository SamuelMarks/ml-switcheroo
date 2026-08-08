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


def test_emit_expression_binary_op():
  """Tests emitting binary operation hits sw.op resolution."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.add"}}}))

  expr = cst.BinaryOperation(left=cst.Integer("1"), operator=cst.Add(), right=cst.Integer("2"))
  val, ops = emitter._emit_expression(expr)
  assert any(op.name == "stablehlo.add" for op in ops)


def test_emit_call_fallback_sw_op():
  """Tests fallback of _emit_call when op is unknown but has sw.op in args."""
  emitter, semantics = setup_emitter()

  # Mock get_definition to return something for the inner op but None for outer
  def mock_def(api_path):
    """Mocks definition resolution."""
    if api_path == "known_inner":
      return ("id", {"variants": {"stablehlo": {"api": "stablehlo.inner"}}})
    return None

  semantics.get_definition = MagicMock(side_effect=mock_def)

  # Outer call is unknown, but inner call is known. Wait, if inner is a call, it's evaluated first?
  # But we want super()._emit_expression to return a sw.op!
  # super()._emit_expression on a binary op returns sw.op.
  # If we pass a binary op as an argument to an unknown call:
  expr = cst.Call(
    func=cst.Name("unknown"),
    args=[cst.Arg(value=cst.BinaryOperation(left=cst.Integer("1"), operator=cst.Add(), right=cst.Integer("2")))],
  )
  val, ops = emitter._emit_call(expr)
  # inner op is Add, which generates sw.op. Since outer is unknown, it falls back to super()._emit_expression
  # which processes the children, returning sw.op, which then gets resolved.
  # Wait, Add resolves to sw.op with name="add". We need get_definition("add") to return something.
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.add"}}}))
  val, ops = emitter._emit_call(expr)
  assert any(op.name == "stablehlo.add" for op in ops)


def test_emit_call_fallback_sw_constant():
  """Tests fallback of _emit_call when op is unknown but has constant args."""
  from ml_switcheroo.core.mlir.cst import OperationNode

  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=None)

  expr = cst.Call(
    func=cst.Name("unknown"),
    args=[cst.Arg(value=cst.Integer("99"))],
  )

  # Mock super()._emit_expression to return a sw.constant directly to hit the unreachable branch
  _original_emit = emitter._emit_expression

  def mock_super_emit(*args, **kwargs):
    """Mocks the superclass _emit_expression call."""
    op = OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value='"1.0"')])
    return "%0", [op]

  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
  import unittest.mock

  with unittest.mock.patch.object(PythonToMlirEmitter, "_emit_expression", side_effect=mock_super_emit):
    val, ops = emitter._emit_call(expr)

  # should have stablehlo.constant in the ops list
  has_constant = any(op.name == "stablehlo.constant" for op in ops)
  assert has_constant


def test_extract_literal():
  """Test function."""
  emitter, _ = setup_emitter()
  assert emitter._extract_literal(cst.Integer("5")) == 5
  assert emitter._extract_literal(cst.Float("5.5")) == 5.5
  assert emitter._extract_literal(cst.SimpleString('"hello"')) == "hello"
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


def test_resolve_nested_sw_constant():
  """Tests resolving nested sw.constant ops in region blocks."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}}))

  # Inject sw.constant inside the lambda body. We simulate this by mocking the call processing
  # where a nested op list is collected.
  # However, it's easier to just call _emit_call with a lambda whose body will generate a sw.constant.
  # If we use a literal inside the lambda, _emit_expression on it will create sw.constant!
  expr = cst.Call(
    func=cst.Name("reduce"),
    args=[
      cst.Arg(value=cst.Lambda(params=cst.Parameters(params=[cst.Param(name=cst.Name("a"))]), body=cst.Integer("42")))
    ],
  )
  val, ops = emitter._emit_call(expr)
  # The lambda block should contain a stablehlo.constant because _resolve_sw_constant was called
  inner_ops = ops[0].regions[0].blocks[0].operations
  assert inner_ops[0].name == "stablehlo.constant"


def test_emit_call_string_attr():
  """Tests emitting call with string attribute."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.custom"}}}))
  # A keyword arg with a boolean/non-string
  expr = cst.Call(
    func=cst.Name("custom"),
    args=[
      cst.Arg(value=cst.Name("False"), keyword=cst.Name("is_true")),
      cst.Arg(value=cst.Integer("42"), keyword=cst.Name("some_int")),
    ],
  )
  val, ops = emitter._emit_call(expr)
  assert ops[0].attributes[0].name == "is_true"
  assert "%error" in ops[0].attributes[0].value
  assert ops[0].attributes[1].name == "some_int"
  assert ops[0].attributes[1].value == "42"
