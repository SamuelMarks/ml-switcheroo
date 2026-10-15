"""Tests for MLIR dialect coverage."""

from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.core.mlir.dialect import OpSchema


def test_opschema_validate_wrong_name() -> None:
  """Verify opschema validate wrong name behavior."""
  schema = OpSchema("my.op")
  node = OperationNode(name="other.op", operands=[], results=[], attributes=[], regions=[])
  assert schema.validate(node) is False


def test_load_official_ops() -> None:
  """Verify load official ops behavior."""
  from ml_switcheroo.core.mlir.official_dialects import OFFICIAL_OPS

  assert len(OFFICIAL_OPS) > 0
