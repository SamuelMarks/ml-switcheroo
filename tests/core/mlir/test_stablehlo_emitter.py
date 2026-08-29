"""Test suite for the Stablehlo Emitter module."""

import typing
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, TypeNode
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, typing.Any] = {}
    abs_def: dict[str, typing.Any] = {"variants": {"torch": {"api": "torch.abs"}, "stablehlo": {"api": "stablehlo.abs"}}}
    self._inject("Abs", "torch.abs", abs_def)
    add_def: dict[str, typing.Any] = {"variants": {"torch": {"api": "torch.add"}, "stablehlo": {"api": "stablehlo.add"}}}
    self._inject("Add", "torch.add", add_def)

  def _inject(self, name: str, api: str, defn: dict[str, typing.Any]) -> None:
    """Mock implementation of  inject."""
    self._reverse_index[api] = (name, defn)

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)


def emit_code(code: str) -> str:
  """Emits code."""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir_node: typing.Any = emitter.convert(tree)
  return typing.cast(str, mlir_node.to_text())


def test_module_structure() -> None:
  """Verifies the behavior of module structure."""
  code: str = "\nclass MyNet:\n    pass\n"
  mlir: str = emit_code(code)
  assert 'module {sym_name = "MyNet"}' in mlir


def test_func_structure_and_types() -> None:
  """Verifies the behavior of function structure and types."""
  code: str = "\ndef forward(x: Tensor, i: int) -> float:\n    return x\n"
  mlir: str = emit_code(code)
  assert "func.func" in mlir
  assert 'sym_name = "forward"' in mlir
  assert "tensor<*xf32>" in mlir
  assert "i32" in mlir
  assert "function_type = (tensor<*xf32>, i32) -> f32" in mlir


def test_func_structure_implicit_returns() -> None:
  """Verifies function type inference without explicit returns."""
  code: str = "\ndef forward(x: Tensor):\n    y = x\n    return y\n"
  mlir: str = emit_code(code)
  assert "function_type = (tensor<*xf32>) -> tensor<*xf32>" in mlir


def test_stablehlo_op_resolution() -> None:
  """Verifies the behavior of StableHLO op resolution."""
  code: str = "y = torch.abs(x)"
  mlir: str = emit_code(code)
  assert "sw.op" not in mlir
  assert "stablehlo.abs" in mlir
  assert ": tensor<*xf32>" in mlir


def test_unknown_op_fallback() -> None:
  """Verifies the behavior of unknown op fallback."""
  code: str = "y = torch.unknown(x)"
  mlir: str = emit_code(code)
  assert "stablehlo" not in mlir
  assert "sw.op" in mlir
  assert 'type = "torch.unknown"' in mlir


def test_return_statement() -> None:
  """Verifies the behavior of return statement."""
  code: str = "return x"
  mlir: str = emit_code(code)
  assert "func.return" in mlir


def test_expression_chaining() -> None:
  """Verifies the behavior of expression chaining."""
  code: str = "y = torch.add(torch.abs(x), x)"
  mlir: str = emit_code(code)
  assert "stablehlo.abs" in mlir
  assert "stablehlo.add" in mlir
  assert mlir.count("=") >= 2


def test_constant_resolution() -> None:
  """Verifies the behavior of stablehlo.constant resolution."""
  code: str = "y = 5.0"
  mlir: str = emit_code(code)
  assert "stablehlo.constant" in mlir
  assert "dense<5.0>" in mlir
  assert "tensor<f32>" in mlir

  code_int: str = "y = 42"
  mlir_int: str = emit_code(code_int)
  assert "stablehlo.constant" in mlir_int
  assert "dense<42>" in mlir_int
  assert "tensor<i32>" in mlir_int


def test_attribute_serialization() -> None:
  """Verifies the behavior of kwargs to attribute serialization."""
  code: str = 'y = torch.convolution(x, w, window_strides=[1, 1], padding="SAME")'

  # Note: The test uses MockSemantics, so we need to inject convolution definition
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  semantics._inject("Convolution", "torch.convolution", {"variants": {"stablehlo": {"api": "stablehlo.convolution"}}})
  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert "stablehlo.convolution" in mlir
  assert "window_strides = dense<[1, 1]> : tensor<2xi64>" in mlir
  assert 'padding = "SAME"' in mlir


def test_conditional_control_flow() -> None:
  """Verifies the behavior of stablehlo.if region generation."""
  code: str = """
def forward(x: Tensor, cond: bool):
    if cond:
        y = torch.abs(x)
    else:
        y = x
    return y
"""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert "stablehlo.if" in mlir
  assert "stablehlo.abs" in mlir
  # Ensure the block structure exists
  assert "{" in mlir
  assert "stablehlo.return" in mlir


def test_conditional_control_flow_no_else() -> None:
  """Verifies stablehlo.if generates an empty else region when missing."""
  code: str = """
def forward(x: Tensor, cond: bool):
    if cond:
        y = torch.abs(x)
    return x
"""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert "stablehlo.if" in mlir
  # Second region should just be the dummy block with a return
  assert mlir.count("stablehlo.return") >= 2


def test_conditional_control_flow_elif() -> None:
  """Verifies elif structure generation."""
  code: str = """
def forward(x: Tensor, cond: bool):
    if cond:
        y = torch.abs(x)
    elif x:
        y = x
    return y
"""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert mlir.count("stablehlo.if") >= 2


def test_while_control_flow() -> None:
  """Verifies the behavior of stablehlo.while region generation."""
  code: str = """
def forward(x: Tensor, count: int):
    while count:
        x = torch.abs(x)
    return x
"""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert "stablehlo.while" in mlir
  assert "stablehlo.abs" in mlir
  # Ensure cond and body regions exist
  assert mlir.count("stablehlo.return") >= 2


def test_higher_order_reduce() -> None:
  """Verifies the behavior of stablehlo.reduce with a lambda."""
  code: str = """
def forward(x: Tensor):
    y = torch.reduce(x, lambda a, b: torch.add(a, b))
    return y
"""
  tree: cst.Module = cst.parse_module(code.strip())
  semantics = MockSemantics()
  # Inject reduce definition
  semantics._inject("Reduce", "torch.reduce", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}})

  emitter = StableHloEmitter(semantics)
  mlir: str = typing.cast(str, emitter.convert(tree).to_text())

  assert "stablehlo.reduce" in mlir
  # Check if regions are properly nested
  assert "{" in mlir
  assert "stablehlo.return" in mlir


# --- Merged from test_stablehlo_emitter_extra.py ---


def setup_emitter() -> tuple[StableHloEmitter, SemanticsManager]:
  """Docstring."""
  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="torch", target_framework="stablehlo")
  ctx = RewriterContext(semantics=semantics, config=config)  # noqa: F841
  emitter = StableHloEmitter(semantics)
  return emitter, semantics


def test_stablehlo_dummy_import() -> None:
  """Docstring."""
  emitter, _ = setup_emitter()
  op: typing.Any = emitter._emit_import(cst.Import(names=[cst.ImportAlias(name=cst.Name("math"))]))
  assert op.name == "stablehlo.dummy_import"


def test_resolve_sw_constant_quotes() -> None:
  """Docstring."""
  emitter, _ = setup_emitter()
  op = OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value='"1.5"')])
  emitter._resolve_sw_constant(op)
  assert op.name == "stablehlo.constant"
  assert op.attributes[0].value == "dense<1.5>"
  assert op.result_types[0].body == "tensor<f32>"


def test_resolve_sw_op_quotes() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.fake"}}}))  # type: ignore
  op = OperationNode(name="sw.op", attributes=[AttributeNode(name="type", value='"torch.fake"')])
  emitter._resolve_sw_op(op)
  assert op.name == "stablehlo.fake"


def test_lookup_stablehlo_op_not_found() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=None)  # type: ignore
  assert emitter._lookup_stablehlo_op("torch.fake") is None


def test_lookup_stablehlo_op_no_variant() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {}}))  # type: ignore
  assert emitter._lookup_stablehlo_op("torch.fake") is None


def test_emit_expression_binary_op() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.add"}}}))  # type: ignore

  expr = cst.BinaryOperation(left=cst.Integer("1"), operator=cst.Add(), right=cst.Integer("2"))
  val: typing.Any
  ops: list[typing.Any]
  val, ops = emitter._emit_expression(expr)  # type: ignore
  assert any(op.name == "stablehlo.add" for op in ops)


def test_emit_call_fallback_sw_op() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()

  # Mock get_definition to return something for the inner op but None for outer
  def mock_def(api_path: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mocks definition resolution."""
    if api_path == "known_inner":
      return ("id", {"variants": {"stablehlo": {"api": "stablehlo.inner"}}})
    return None

  semantics.get_definition = MagicMock(side_effect=mock_def)  # type: ignore

  # Outer call is unknown, but inner call is known. Wait, if inner is a call, it's evaluated first?
  # But we want super()._emit_expression to return a sw.op!
  # super()._emit_expression on a binary op returns sw.op.
  # If we pass a binary op as an argument to an unknown call:
  expr = cst.Call(
    func=cst.Name("unknown"),
    args=[cst.Arg(value=cst.BinaryOperation(left=cst.Integer("1"), operator=cst.Add(), right=cst.Integer("2")))],
  )
  val: typing.Any
  ops: list[typing.Any]
  val, ops = emitter._emit_call(expr)
  # inner op is Add, which generates sw.op. Since outer is unknown, it falls back to super()._emit_expression
  # which processes the children, returning sw.op, which then gets resolved.
  # Wait, Add resolves to sw.op with name="add". We need get_definition("add") to return something.
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.add"}}}))  # type: ignore
  val, ops = emitter._emit_call(expr)
  assert any(op.name == "stablehlo.add" for op in ops)


def test_emit_call_fallback_sw_constant() -> None:
  """Docstring."""
  from ml_switcheroo.core.mlir.cst import OperationNode

  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=None)  # type: ignore

  expr = cst.Call(
    func=cst.Name("unknown"),
    args=[cst.Arg(value=cst.Integer("99"))],
  )

  # Mock super()._emit_expression to return a sw.constant directly to hit the unreachable branch
  _original_emit = emitter._emit_expression

  def mock_super_emit(*args: typing.Any, **kwargs: typing.Any) -> tuple[str, list[OperationNode]]:
    """Mocks the superclass _emit_expression call."""
    op = OperationNode(name="sw.constant", attributes=[AttributeNode(name="value", value='"1.0"')])
    return "%0", [op]

  import unittest.mock

  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  with unittest.mock.patch.object(PythonToMlirEmitter, "_emit_expression", side_effect=mock_super_emit):
    val: typing.Any
    ops: list[typing.Any]
    val, ops = emitter._emit_call(expr)

  # should have stablehlo.constant in the ops list
  has_constant = any(op.name == "stablehlo.constant" for op in ops)
  assert has_constant


def test_extract_literal() -> None:
  """Docstring."""
  emitter, _ = setup_emitter()
  assert emitter._extract_literal(cst.Integer("5")) == 5
  assert emitter._extract_literal(cst.Float("5.5")) == 5.5
  assert emitter._extract_literal(cst.SimpleString('"hello"')) == "hello"
  assert emitter._extract_literal(cst.List(elements=[cst.Element(value=cst.Integer("1"))])) == [1]
  assert emitter._extract_literal(cst.Tuple(elements=[cst.Element(value=cst.Integer("2"))])) == [2]
  assert emitter._extract_literal(cst.Pass()) == "%error"


def test_emit_call_lambda() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}}))  # type: ignore
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
  val: typing.Any
  ops: list[typing.Any]
  val, ops = emitter._emit_call(expr)
  assert ops[0].name == "stablehlo.reduce"
  assert len(ops[0].regions) == 1


def test_resolve_nested_sw_constant() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}}))  # type: ignore

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
  val: typing.Any
  ops: list[typing.Any]
  val, ops = emitter._emit_call(expr)
  # The lambda block should contain a stablehlo.constant because _resolve_sw_constant was called
  inner_ops = ops[0].regions[0].blocks[0].operations
  assert inner_ops[0].name == "stablehlo.constant"


def test_emit_call_string_attr() -> None:
  """Docstring."""
  emitter, semantics = setup_emitter()
  semantics.get_definition = MagicMock(return_value=("id", {"variants": {"stablehlo": {"api": "stablehlo.custom"}}}))  # type: ignore
  # A keyword arg with a boolean/non-string
  expr = cst.Call(
    func=cst.Name("custom"),
    args=[
      cst.Arg(value=cst.Name("False"), keyword=cst.Name("is_true")),
      cst.Arg(value=cst.Integer("42"), keyword=cst.Name("some_int")),
    ],
  )
  val: typing.Any
  ops: list[typing.Any]
  val, ops = emitter._emit_call(expr)
  assert ops[0].attributes[0].name == "is_true"
  assert "%error" in ops[0].attributes[0].value
  assert ops[0].attributes[1].name == "some_int"
  assert ops[0].attributes[1].value == "42"


# --- Merged from test_stablehlo_emitter_extra2.py ---


def test_resolve_sw_op_no_type_attr() -> None:
  """Docstring."""
  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)
  op = OperationNode(name="sw.op", operands=[], attributes=[])
  emitter._resolve_sw_op(op)
  assert op.name == "sw.op"  # Should not be modified


def test_map_py_type_to_mlir() -> None:
  """Docstring."""
  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)
  res: str = emitter._map_py_type_to_mlir("float")
  assert res == "f32"


# --- Merged from test_stablehlo_emitter_extra3.py ---


def test_while_return() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "while True:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_while(typing.cast(cst.While, tree.body[0]))


def test_if_true_return() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_if_else_return() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  pass\nelse:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_if_elif_return() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "if True:\n  pass\nelif False:\n  return x"
  tree: cst.Module = cst.parse_module(code)
  emitter._emit_if(typing.cast(cst.If, tree.body[0]))


def test_sw_constant_existing_type() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  op = OperationNode(name="sw.constant", attributes=[], result_types=[TypeNode("tensor<f32>")])
  import ml_switcheroo.core.mlir.cst as m_cst

  op.attributes.append(m_cst.AttributeNode("value", "5.0"))
  emitter._resolve_sw_constant(op)


def test_sw_op_existing_type() -> None:
  """Docstring."""
  mgr = MagicMock()
  mgr.get_definition.return_value = ("torch", {"variants": {"jax": {"api": "jax.numpy.abs"}}})
  emitter = StableHloEmitter(mgr)
  op = OperationNode(name="sw.op", attributes=[], result_types=[TypeNode("tensor<f32>")])
  import ml_switcheroo.core.mlir.cst as m_cst

  op.attributes.append(m_cst.AttributeNode("type", '"torch.abs"'))
  emitter._lookup_stablehlo_op = MagicMock(return_value="stablehlo.abs")  # type: ignore
  emitter._resolve_sw_op(op)


def test_call_non_name() -> None:
  """Docstring."""
  emitter = StableHloEmitter(MagicMock())
  code: str = "(lambda x: x)()"
  tree: cst.Module = cst.parse_module(code)
  expr: typing.Any = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, tree.body[0]).body[0]).value
  emitter._emit_call(expr)


def test_func_def_multiple_params() -> None:
  """Docstring."""
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
