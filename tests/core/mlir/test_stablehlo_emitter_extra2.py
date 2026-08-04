"""Test module."""

from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_resolve_sw_op_no_type_attr():
  """Test function."""
  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)
  op = OperationNode(name="sw.op", operands=[], attributes=[])
  emitter._resolve_sw_op(op)
  assert op.name == "sw.op"  # Should not be modified


def test_map_py_type_to_mlir():
  """Test function."""
  semantics = SemanticsManager()
  emitter = StableHloEmitter(semantics)
  res = emitter._map_py_type_to_mlir("float")
  assert res == "f32"
