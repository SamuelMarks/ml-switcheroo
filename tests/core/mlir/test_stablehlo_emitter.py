"""Test suite for the Stablehlo Emitter module."""

import typing
import libcst as cst
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

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
