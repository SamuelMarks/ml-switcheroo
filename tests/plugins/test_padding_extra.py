"""Test suite for the Padding Extra module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.core.hooks import HookContext


def test_padding_no_conf():
  """Verifies the behavior of padding no conf."""
  node = cst.Call(func=cst.Name("pad"))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {}
  res = transform_padding(node, ctx)
  assert res is node
