"""Unit tests for `StableHloEmitter` (`ml_switcheroo.core.mlir.stablehlo_emitter`)."""

import typing
from unittest import mock

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, TypeNode, ValueNode
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


class _StubSemantics(SemanticsManager):
  """Lightweight in-memory `SemanticsManager` mapping Torch ops to StableHLO dialect ops."""

  def __init__(self) -> None:
    """Populate `Abs` and `Add` mappings without disk I/O."""
    self.data = {}
    self._reverse_index: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.inject("Abs", "torch.abs", {"variants": {"torch": {"api": "torch.abs"}, "stablehlo": {"api": "stablehlo.abs"}}})
    self.inject("Add", "torch.add", {"variants": {"torch": {"api": "torch.add"}, "stablehlo": {"api": "stablehlo.add"}}})
    self.inject("BinAdd", "binop.add", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})

  def resolve_operation(self, api_name: str) -> typing.Never:
    """Raise `AttributeError` to exercise legacy `get_definition` fallback on custom providers.

    Args:
        api_name: Qualified API string.

    Returns:
        Never returns; raises `AttributeError`.

    Raises:
        AttributeError: Always raised to trigger `get_definition` fallback.
    """
    raise AttributeError(api_name)

  def inject(self, name: str, api: str, defn: dict[str, JsonValue]) -> None:
    """Register an operation mapping in `_reverse_index`.

    Args:
        name: Abstract operation identifier.
        api: Source framework API string.
        defn: Operation definition dictionary.
    """
    self._reverse_index[api] = (name, defn)

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up `(op_name, definition)` by source API name.

    Args:
        name: Qualified API string.

    Returns:
        Matching `(op_name, definition)` tuple or `None`.
    """
    return self._reverse_index.get(name)


def emit_code(code: str, semantics: SemanticsManager | None = None) -> str:
  """Parse Python `code` and lower it to StableHLO text.

  Args:
      code: Python source snippet.
      semantics: Optional `SemanticsManager` override.

  Returns:
      Serialized StableHLO text string.
  """
  sem = semantics if semantics is not None else _StubSemantics()
  return StableHloEmitter(sem).convert(cst.parse_module(code.strip())).to_text()


def test_module_and_function_signatures() -> None:
  """Verify class `module` lowering, `func.func` type annotations, inferred returns, and untyped params."""
  assert 'module {sym_name = "MyNet"}' in emit_code(f"class MyNet:{_NL}    pass")

  typed_fn = emit_code(f"def forward(x: Tensor, i: int) -> float:{_NL}    return x")
  assert "func.func" in typed_fn
  assert 'sym_name = "forward"' in typed_fn
  assert "function_type = (tensor<*xf32>, i32) -> f32" in typed_fn

  inferred_fn = emit_code(f"def forward(x: Tensor):{_NL}    y = x{_NL}    return y")
  assert "function_type = (tensor<*xf32>) -> tensor<*xf32>" in inferred_fn

  void_fn = emit_code(f"def noop(a, b):{_NL}    return")
  assert 'sym_name = "noop"' in void_fn

  # Defensive branch when `param.name` is not a `cst.Name`
  emitter = StableHloEmitter(_StubSemantics())
  parsed_fn = cst.ensure_type(cst.parse_statement("def f(x): pass"), cst.FunctionDef)
  with mock.patch.object(cst.Param, "name", cst.Tuple([])):
    assert emitter._emit_func_def(parsed_fn).name == "func.func"


def test_ops_constants_attributes_and_imports() -> None:
  """Verify StableHLO op resolution, unknown fallbacks, constants, attributes, and ignored imports."""
  assert "stablehlo.abs" in emit_code("y = torch.abs(x)")
  assert 'type = "torch.unknown"' in emit_code("y = torch.unknown(x)")
  assert "func.return" in emit_code("return x")

  chained = emit_code("y = torch.add(torch.abs(x), x)")
  assert "stablehlo.abs" in chained
  assert "stablehlo.add" in chained

  float_mlir = emit_code("y = 5.0")
  assert "dense<5.0>" in float_mlir
  assert "tensor<f32>" in float_mlir

  int_mlir = emit_code("y = 42")
  assert "dense<42>" in int_mlir
  assert "tensor<i32>" in int_mlir

  sem = _StubSemantics()
  sem.inject("Convolution", "torch.convolution", {"variants": {"stablehlo": {"api": "stablehlo.convolution"}}})
  conv_mlir = emit_code('y = torch.convolution(x, w, window_strides=[1, 1], padding="SAME")', sem)
  assert "stablehlo.convolution" in conv_mlir
  assert "window_strides = dense<[1, 1]> : tensor<2xi64>" in conv_mlir
  assert 'padding = "SAME"' in conv_mlir

  # Imports produce `stablehlo.dummy_import` directly and are stripped at module level
  emitter = StableHloEmitter(sem)
  assert emitter._emit_import(cst.Import(names=[cst.ImportAlias(name=cst.Name("math"))])).name == "stablehlo.dummy_import"
  assert emitter.convert(cst.parse_module(f"import x{_NL}from y import z")).body.operations == []


def test_control_flow_if_elif_else_and_while() -> None:
  """Verify `stablehlo.if` and `stablehlo.while` region synthesis across return and non-return branches."""
  if_else = emit_code(
    _NL.join(
      [
        "def forward(x: Tensor, cond: bool):",
        "    if cond:",
        "        y = torch.abs(x)",
        "    else:",
        "        y = x",
        "    return y",
      ]
    )
  )
  assert "stablehlo.if" in if_else
  assert "stablehlo.return" in if_else

  if_no_else = emit_code(
    _NL.join(
      [
        "def forward(x: Tensor, cond: bool):",
        "    if cond:",
        "        return x",
        "    return x",
      ]
    )
  )
  assert if_no_else.count("stablehlo.return") >= 1

  if_elif = emit_code(
    _NL.join(
      [
        "def forward(x: Tensor, cond: bool):",
        "    if cond:",
        "        pass",
        "    elif x:",
        "        return x",
        "    else:",
        "        return x",
      ]
    )
  )
  assert if_elif.count("stablehlo.if") >= 2

  while_mlir = emit_code(
    _NL.join(
      [
        "def forward(x: Tensor, count: int):",
        "    while count:",
        "        x = torch.abs(x)",
        "    while count:",
        "        pass",
        "    while count:",
        "        return x",
        "    return x",
      ]
    )
  )
  assert while_mlir.count("stablehlo.while") == 3

  # Invalid `orelse`, empty `elif` list, and returning `elif` list defensive branches
  emitter = StableHloEmitter(_StubSemantics())
  base_if = cst.ensure_type(cst.parse_module(f"if True: pass{_NL}").body[0], cst.If)
  with mock.patch.object(cst.If, "orelse", cst.Pass()):
    assert emitter._emit_if(base_if)[-1].name == "stablehlo.if"

  elif_node = cst.ensure_type(cst.parse_module(f"if True: pass{_NL}elif False: pass").body[0], cst.If)
  orig_emit_if = emitter._emit_if
  call_count = 0

  def _empty_nested_if(node: cst.If) -> list[OperationNode]:
    """Return empty ops on recursive `elif` invocation.

    Args:
        node: `cst.If` node being lowered.

    Returns:
        Operation list or `[]` for nested `elif`.
    """
    nonlocal call_count
    call_count += 1
    return [] if call_count > 1 else orig_emit_if(node)

  with mock.patch.object(emitter, "_emit_if", side_effect=_empty_nested_if):
    assert emitter._emit_if(elif_node)[-1].regions[1].blocks[0].operations[-1].name == "stablehlo.return"

  ret_count = 0

  def _ret_nested_if(node: cst.If) -> list[OperationNode]:
    """Return `stablehlo.return` on recursive `elif` invocation.

    Args:
        node: `cst.If` node being lowered.

    Returns:
        Operation list ending in `stablehlo.return` for nested `elif`.
    """
    nonlocal ret_count
    ret_count += 1
    return [OperationNode(name="stablehlo.return")] if ret_count > 1 else orig_emit_if(node)

  with mock.patch.object(emitter, "_emit_if", side_effect=_ret_nested_if):
    assert emitter._emit_if(elif_node)[-1].regions[1].blocks[0].operations[-1].name == "stablehlo.return"


def test_higher_order_lambdas_literals_and_resolution_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `stablehlo.reduce` lambdas, `_extract_literal`, `_resolve_sw_*`, and fallback calls.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  sem = _StubSemantics()
  sem.inject("Reduce", "torch.reduce", {"variants": {"stablehlo": {"api": "stablehlo.reduce"}}})
  sem.inject("Custom", "custom", {"variants": {"stablehlo": {"api": "stablehlo.custom"}}})
  sem.inject("BinAdd", "add", {"variants": {"stablehlo": {"api": "stablehlo.add"}}})
  sem.inject("NoStable", "torch.nostable", {"variants": {"jax": {"api": "jax.numpy.abs"}}})

  reduce_mlir = emit_code(
    _NL.join(
      [
        "def forward(x: Tensor):",
        "    y = torch.reduce(x, lambda a, b: torch.add(a, 42))",
        "    return y",
      ]
    ),
    sem,
  )
  assert "stablehlo.reduce" in reduce_mlir
  assert "stablehlo.constant" in reduce_mlir

  emitter = StableHloEmitter(sem)
  assert emitter._extract_literal(cst.Integer("5")) == 5
  assert emitter._extract_literal(cst.Float("5.5")) == 5.5
  assert emitter._extract_literal(cst.SimpleString('"hello"')) == "hello"
  assert emitter._extract_literal(cst.List(elements=[cst.Element(value=cst.Integer("1"))])) == [1]
  assert emitter._extract_literal(cst.Tuple(elements=[cst.Element(value=cst.Integer("2"))])) == [2]
  assert emitter._extract_literal(cst.Pass()) == "%error"

  assert emitter._map_py_type_to_mlir("float") == "f32"
  assert emitter._map_py_type_to_mlir("bool") == "i1"
  assert emitter._map_py_type_to_mlir("custom_object") == "!sw.unknown"
  assert emitter._lookup_stablehlo_op("unmapped.op") is None
  assert emitter._lookup_stablehlo_op("torch.nostable") is None

  real_emitter = StableHloEmitter(SemanticsManager())
  assert real_emitter._lookup_stablehlo_op("unmapped.op") is None
  assert real_emitter._lookup_stablehlo_op("torch.abs") == "stablehlo.abs"

  # Keyword args with non-literal boolean vs integer literal
  kw_call = cst.ensure_type(cst.parse_expression("custom(is_true=False, some_int=42)"), cst.Call)
  _, kw_ops = emitter._emit_call(kw_call)
  assert kw_ops[0].attributes[0].name == "is_true"
  assert "%error" in str(kw_ops[0].attributes[0].value)
  assert kw_ops[0].attributes[1].value == "42"

  # Binary op expression resolution and unknown outer call with binary inner arg + non-name callee
  _, bin_ops = emitter._emit_expression(cst.BinaryOperation(cst.Integer("1"), cst.Add(), cst.Integer("2")))
  assert any(op.name == "stablehlo.add" for op in bin_ops)

  unk_outer = cst.Call(
    func=cst.Name("unknown_fn"),
    args=[cst.Arg(value=cst.BinaryOperation(cst.Integer("1"), cst.Add(), cst.Integer("2")))],
  )
  _, unk_ops = emitter._emit_call(unk_outer)
  assert any(op.name == "stablehlo.add" for op in unk_ops)

  lam_call = cst.ensure_type(cst.parse_expression("(lambda x: x)()"), cst.Call)
  _, lam_ops = emitter._emit_call(lam_call)
  assert isinstance(lam_ops, list)

  # Direct `sw.constant` fallback in `super()._emit_expression`
  def _fake_super_emit(_self: PythonToMlirEmitter, _expr: cst.BaseExpression) -> tuple[ValueNode, list[OperationNode]]:
    """Return a synthetic `sw.constant` op from `PythonToMlirEmitter._emit_expression`.

    Args:
        _self: Emitter instance.
        _expr: Expression node.

    Returns:
        `(ValueNode, [OperationNode])` pair containing an unresolved `sw.constant`.
    """
    from ml_switcheroo.core.mlir.cst import ValueNode as VNode

    return VNode(name="%0"), [OperationNode(name="sw.constant", attributes=[AttributeNode("value", '"1.0"')])]

  monkeypatch.setattr(PythonToMlirEmitter, "_emit_expression", _fake_super_emit)
  _, const_ops = emitter._emit_call(cst.ensure_type(cst.parse_expression("unknown_fn(99)"), cst.Call))
  assert any(op.name == "stablehlo.constant" for op in const_ops)

  # `_resolve_sw_constant` and `_resolve_sw_op` edge cases (missing attrs, pre-populated result_types)
  op_c_empty = OperationNode(name="sw.constant")
  emitter._resolve_sw_constant(op_c_empty)
  assert op_c_empty.name == "stablehlo.constant"
  assert op_c_empty.result_types == []

  op_c_quotes = OperationNode(name="sw.constant", attributes=[AttributeNode("value", '"1.5"')])
  emitter._resolve_sw_constant(op_c_quotes)
  assert op_c_quotes.name == "stablehlo.constant"
  assert op_c_quotes.result_types[0].body == "tensor<f32>"

  op_c_typed = OperationNode(
    name="sw.constant",
    attributes=[AttributeNode("value", "5.0")],
    result_types=[TypeNode("tensor<i64>")],
  )
  emitter._resolve_sw_constant(op_c_typed)
  assert op_c_typed.result_types[0].body == "tensor<i64>"

  op_sw_empty = OperationNode(name="sw.op")
  emitter._resolve_sw_op(op_sw_empty)
  assert op_sw_empty.name == "sw.op"

  op_sw_typed = OperationNode(
    name="sw.op",
    attributes=[AttributeNode("type", '"torch.abs"')],
    result_types=[TypeNode("tensor<i64>")],
  )
  emitter._resolve_sw_op(op_sw_typed)
  assert op_sw_typed.name == "stablehlo.abs"
  assert op_sw_typed.result_types[0].body == "tensor<i64>"

  # `StableHloParser` smoke roundtrip
  parsed_mod = StableHloParser("module {}").parse()
  assert len(parsed_mod.body.operations) == 1
  assert parsed_mod.body.operations[0].name == "module"
