"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.loop_unroll import transform_loops, _analyze_range_iterator
from typing import List


def test_transform_loops_not_required() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = False

  code: str = "for i in range(10):\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.BaseStatement = module.body[0]

  transformed: cst.CSTNode = transform_loops(for_node, ctx)
  assert transformed is for_node  # unchanged


def test_transform_loops_range() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_fw = "jax"

  code: str = "for i in range(10):\n    x += 1"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.BaseStatement = module.body[0]

  transformed: cst.CSTNode = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for range
  first_node: cst.CSTNode = transformed.nodes[0]
  assert hasattr(first_node, "leading_lines")
  comments: List[str] = [
    line.comment.value for line in getattr(first_node, "leading_lines") if getattr(line, "comment", None)
  ]
  assert any("requires explicit functional loops" in c for c in comments)


def test_transform_loops_not_range() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_fw = "jax"

  code: str = "for i in [1, 2, 3]:\n    x += i"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.BaseStatement = module.body[0]

  transformed: cst.CSTNode = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for not-range
  first_node: cst.CSTNode = transformed.nodes[0]
  assert hasattr(first_node, "leading_lines")
  comments: List[str] = [
    line.comment.value for line in getattr(first_node, "leading_lines") if getattr(line, "comment", None)
  ]
  assert any("requires structural rewrite" in c for c in comments)


def test_analyze_range_iterator_not_call() -> None:
  """Docstring."""
  code: str = "for i in my_list:\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.CSTNode = module.body[0]

  is_range: bool
  args: List[cst.Arg]
  is_range, args = _analyze_range_iterator(getattr(for_node, "iter"))
  assert not is_range
  assert args == []


def test_analyze_range_iterator_not_range_name() -> None:
  """Docstring."""
  code: str = "for i in my_func(10):\n    pass"
  module: cst.Module = cst.parse_module(code)
  for_node: cst.CSTNode = module.body[0]

  is_range: bool
  args: List[cst.Arg]
  is_range, args = _analyze_range_iterator(getattr(for_node, "iter"))
  assert not is_range
  assert args == []
