"""Test suite for plugin."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.loop_unroll import _analyze_range_iterator, transform_loops


def test_transform_loops_not_required() -> None:
  """Verify `ensure_type` handles transform loops not required as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = False

  code: str = "for i in range(10):\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.For = cst.ensure_type(module.body[0], cst.For)

  transformed: cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement] = transform_loops(for_node, ctx)
  assert transformed is for_node  # unchanged


def test_transform_loops_range() -> None:
  """Verify `ensure_type` returns `2` for transform loops range."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_framework = "jax"

  code: str = "for i in range(10):\n    x += 1"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.For = cst.ensure_type(module.body[0], cst.For)

  transformed: cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement] = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for range
  first_node: cst.BaseStatement = transformed.nodes[0]
  assert isinstance(first_node, (cst.SimpleStatementLine, cst.BaseCompoundStatement))
  comments: list[str] = [line.comment.value for line in first_node.leading_lines if line.comment is not None]
  assert any("requires explicit functional loops" in c for c in comments)


def test_transform_loops_not_range() -> None:
  """Verify `ensure_type` returns `2` for transform loops not range."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_framework = "jax"

  code: str = "for i in [1, 2, 3]:\n    x += i"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.For = cst.ensure_type(module.body[0], cst.For)

  transformed: cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement] = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for not-range
  first_node: cst.BaseStatement = transformed.nodes[0]
  assert isinstance(first_node, (cst.SimpleStatementLine, cst.BaseCompoundStatement))
  comments: list[str] = [line.comment.value for line in first_node.leading_lines if line.comment is not None]
  assert any("requires structural rewrite" in c for c in comments)


def test_analyze_range_iterator_not_call() -> None:
  """Verify `_analyze_range_iterator` handles not call as expected."""
  code: str = "for i in my_list:\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.For = cst.ensure_type(module.body[0], cst.For)

  is_range: bool
  args: list[cst.Arg]
  is_range, args = _analyze_range_iterator(for_node.iter)
  assert not is_range
  assert args == []


def test_analyze_range_iterator_not_range_name() -> None:
  """Verify `_analyze_range_iterator` handles not range name as expected."""
  code: str = "for i in my_func(10):\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.For = cst.ensure_type(module.body[0], cst.For)

  is_range: bool
  args: list[cst.Arg]
  is_range, args = _analyze_range_iterator(for_node.iter)
  assert not is_range
  assert args == []
