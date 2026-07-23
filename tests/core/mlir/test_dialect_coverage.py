"""Tests for MLIR dialect coverage."""

from ml_switcheroo.core.mlir.dialect import OpSchema
from ml_switcheroo.core.mlir.nodes import OperationNode


def test_opschema_validate_wrong_name():
  """Test opschema validate wrong name."""
  schema = OpSchema("my.op")
  node = OperationNode(name="other.op", operands=[], results=[], attributes=[], regions=[])
  assert schema.validate(node) is False
