"""Test suite for the Method Property Extra module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.method_property import transform_method_to_property
from ml_switcheroo.core.hooks import HookContext


def test_method_property_not_attribute():
  """Verifies the behavior of method property not attribute."""
  node = cst.Call(func=cst.Name("size"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = transform_method_to_property(node, ctx)
  assert res is node


def test_method_property_unknown_method():
  """Verifies the behavior of method property unknown method."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("unknown")))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = transform_method_to_property(node, ctx)
  assert res is node
