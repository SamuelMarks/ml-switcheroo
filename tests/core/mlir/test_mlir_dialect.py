"""Test suite for the Mlir Dialect module."""

from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode, RegionNode, ValueNode
from ml_switcheroo.core.mlir.dialect import DialectRegistry


def test_valid_module():
  """Verifies the behavior of valid module."""
  op = OperationNode(name="sw.module", attributes=[AttributeNode("sym_name", '"MyMod"')], regions=[RegionNode()])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_module_no_name():
  """Verifies the behavior of invalid module no name."""
  op = OperationNode(name="sw.module", attributes=[], regions=[RegionNode()])
  assert DialectRegistry.validate_op(op) is False


def test_valid_func():
  """Verifies the behavior of valid function."""
  op = OperationNode(name="sw.func", attributes=[AttributeNode("sym_name", '"f"')], regions=[RegionNode()])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_func_no_region():
  """Verifies the behavior of invalid function no region."""
  op = OperationNode(name="sw.func", attributes=[AttributeNode("sym_name", '"f"')], regions=[])
  assert DialectRegistry.validate_op(op) is False


def test_valid_op_instantiation():
  """Verifies the behavior of valid op instantiation."""
  op = OperationNode(name="sw.op", results=[ValueNode("%0")], attributes=[AttributeNode("type", '"Linear"')])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_op_no_result():
  """Verifies the behavior of invalid op no result."""
  op = OperationNode(name="sw.op", results=[], attributes=[AttributeNode("type", '"Linear"')])
  assert DialectRegistry.validate_op(op) is False


def test_unknown_sw_op():
  """Verifies the behavior of unknown sw op."""
  op = OperationNode(name="sw.magic")
  assert DialectRegistry.validate_op(op) is False


def test_external_dialect_allowed():
  """Verifies the behavior of external dialect allowed."""
  op = OperationNode(name="std.add")
  assert DialectRegistry.validate_op(op) is True


def test_abstract_mapping():
  """Verifies the behavior of abstract mapping."""
  assert DialectRegistry.get_abstract_op("Linear") == "sw.op"
