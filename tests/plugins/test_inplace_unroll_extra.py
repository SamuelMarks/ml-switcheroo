"""Test suite for inplace unroll extra coverage."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.plugins.inplace_unroll import unroll_inplace_ops, _get_receiver_name, _get_method_name
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig


def test_inplace_unroll_not_attribute():
  """Test inplace unroll when the node is not an attribute."""
  node = cst.parse_expression("add_(x)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(node, ctx)
  assert res is node


def test_get_receiver_name_not_attribute():
  """Test getting the receiver name when it is not an attribute."""
  node = cst.parse_expression("add_(x)")
  assert _get_receiver_name(node) is None


def test_get_method_name_not_attribute():
  """Test getting the method name when it is not an attribute."""
  node = cst.parse_expression("add_(x)")
  assert _get_method_name(node) is None


def test_inplace_unroll_just_underscore():
  """Test inplace unroll when the method name is just an underscore."""
  node = cst.parse_expression("x._(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(node, ctx)
  assert res is node


def test_inplace_unroll_no_underscore():
  """Test inplace unroll when the method name does not end with an underscore."""
  node = cst.parse_expression("x.add(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(node, ctx)
  assert res is node
