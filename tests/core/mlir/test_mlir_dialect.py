"""
Tests for Switcheroo Dialect Schema Validation.

Verifies:
1. Valid ops pass validation.
2. Missing attributes fail validation.
3. Incorrect region counts fail validation.
4. Unknown dialect ops fail if in 'sw' namespace.
"""

import pytest
from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode, RegionNode, ValueNode
from ml_switcheroo.core.mlir.dialect import DialectRegistry


def test_valid_module():
  op = OperationNode(name="sw.module", attributes=[AttributeNode("sym_name", '"MyMod"')], regions=[RegionNode()])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_module_no_name():
  op = OperationNode(
    name="sw.module",
    attributes=[],  # Missing sym_name
    regions=[RegionNode()],
  )
  assert DialectRegistry.validate_op(op) is False


def test_valid_func():
  op = OperationNode(name="sw.func", attributes=[AttributeNode("sym_name", '"f"')], regions=[RegionNode()])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_func_no_region():
  op = OperationNode(
    name="sw.func",
    attributes=[AttributeNode("sym_name", '"f"')],
    regions=[],  # Missing body
  )
  assert DialectRegistry.validate_op(op) is False


def test_valid_op_instantiation():
  op = OperationNode(name="sw.op", results=[ValueNode("%0")], attributes=[AttributeNode("type", '"Linear"')])
  assert DialectRegistry.validate_op(op) is True


def test_invalid_op_no_result():
  # sw.op implies creating something (e.g. a layer instance)
  op = OperationNode(name="sw.op", results=[], attributes=[AttributeNode("type", '"Linear"')])
  assert DialectRegistry.validate_op(op) is False


def test_unknown_sw_op():
  op = OperationNode(name="sw.magic")
  assert DialectRegistry.validate_op(op) is False


def test_external_dialect_allowed():
  # 'std.add' is not in our registry, but doesn't start with sw., so we pass it
  op = OperationNode(name="std.add")
  assert DialectRegistry.validate_op(op) is True


def test_abstract_mapping():
  assert DialectRegistry.get_abstract_op("Linear") == "sw.op"
