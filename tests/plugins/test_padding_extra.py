"""Test suite for the Padding Extra module."""

import libcst as cst
from typing import Union
from unittest.mock import MagicMock
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.core.hooks import HookContext


def test_padding_no_conf() -> None:
  """Verifies the behavior of padding no conf."""
  node: cst.Call = cst.Call(func=cst.Name("pad"))
  semantics: MagicMock = MagicMock()
  ctx: HookContext = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {}
  res: Union[cst.CSTNode, cst.Call] = transform_padding(node, ctx)
  assert res is node
