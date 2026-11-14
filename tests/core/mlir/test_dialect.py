"""Unit tests for MLIR dialect schemas and registry (`ml_switcheroo.core.mlir.dialect`)."""

from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode, RegionNode, ValueNode
from ml_switcheroo.core.mlir.dialect import DialectRegistry, OpSchema
from ml_switcheroo.core.mlir.official_dialects import OFFICIAL_OPS


def test_opschema_and_official_dialects() -> None:
  """Verify `OpSchema.validate_node` rejects mismatched op names and `OFFICIAL_OPS` is populated."""
  schema = OpSchema("my.op", num_regions=1)
  assert schema.validate_node(OperationNode(name="other.op")) is False
  assert len(OFFICIAL_OPS) > 0


def test_dialect_registry_validation_and_mapping() -> None:
  """Verify `DialectRegistry.validate_op` and `get_abstract_op` across `sw.*` and external ops."""
  valid_mod = OperationNode(
    name="sw.module",
    attributes=[AttributeNode(name="sym_name", value='"MyMod"')],
    regions=[RegionNode()],
  )
  assert DialectRegistry.validate_op(valid_mod) is True
  assert DialectRegistry.validate_op(OperationNode(name="sw.module", regions=[RegionNode()])) is False

  valid_fn = OperationNode(
    name="sw.func",
    attributes=[AttributeNode(name="sym_name", value='"f"')],
    regions=[RegionNode()],
  )
  assert DialectRegistry.validate_op(valid_fn) is True
  assert (
    DialectRegistry.validate_op(OperationNode(name="sw.func", attributes=[AttributeNode("sym_name", '"f"')])) is False
  )

  valid_op = OperationNode(
    name="sw.op",
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="type", value='"Linear"')],
  )
  assert DialectRegistry.validate_op(valid_op) is True
  assert DialectRegistry.validate_op(OperationNode(name="sw.op", attributes=[AttributeNode("type", '"Linear"')])) is False

  assert DialectRegistry.validate_op(OperationNode(name="sw.magic")) is False
  assert DialectRegistry.validate_op(OperationNode(name="std.add")) is True
  assert DialectRegistry.get_abstract_op("Linear") == "sw.op"
