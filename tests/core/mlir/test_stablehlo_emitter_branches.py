"""Test module."""

import libcst as cst
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_stablehlo_branches():
  """Test function."""
  semantics = SemanticsManager()

  # Mock some behavior
  def mock_get_def(name):
    """Mocks get_definition."""
    return ("dummy", {"variants": {}})

  semantics.get_definition = mock_get_def

  emitter = StableHloEmitter(semantics)

  # 87->91 (if param.annotation is None)
  # 110->115 (elif infer_pass.return_types where it is empty)
  code = "def foo(x):\n    pass"
  tree = cst.parse_module(code)
  emitter.convert(tree)

  # 155->161 (if without else) -> tested in no_else? Wait, if getattr(node, 'orelse', None) is None.
  # We already have test_conditional_control_flow_no_else

  # 193->200 (return without value)
  code = "def foo():\n    return"
  tree = cst.parse_module(code)
  emitter.convert(tree)

  # 218->224 (op.name != sw.op and != sw.constant)
  # 215->232 (empty ops loop?) - it will happen with some expr
  code = "def foo():\n    x = 1"
  tree = cst.parse_module(code)
  emitter.convert(tree)

  # 248->253 (sw.constant without value attr)
  op = OperationNode(name="sw.constant", operands=[], attributes=[])
  emitter._resolve_sw_constant(op)

  # 278->280 (sw.op without type attr)
  op = OperationNode(name="sw.op", operands=[], attributes=[])
  emitter._resolve_sw_op(op)

  # 319->exit (already hit in my extra test? but we need generator or something?)
  # 331->exit
  # 357->exit

  # 407->410 (if not stablehlo_name -> already hit when it's unknown)

  # Let's try _emit_statement with something that gives empty ops or dummy import
  code = "import x\nfrom y import z"
  tree = cst.parse_module(code)
  emitter.convert(tree)
