"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.loop_unroll import transform_loops, _analyze_range_iterator


def test_transform_loops_not_required():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = False

  code = "for i in range(10):\n    pass"
  module = cst.parse_module(code)
  for_node = module.body[0]

  transformed = transform_loops(for_node, ctx)
  assert transformed is for_node  # unchanged


def test_transform_loops_range():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_fw = "jax"

  code = "for i in range(10):\n    x += 1"
  module = cst.parse_module(code)
  for_node = module.body[0]

  transformed = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for range
  first_node = transformed.nodes[0]
  comments = [line.comment.value for line in first_node.leading_lines if line.comment]
  assert any("requires explicit functional loops" in c for c in comments)


def test_transform_loops_not_range():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_control_flow = True
  ctx.target_fw = "jax"

  code = "for i in [1, 2, 3]:\n    x += i"
  module = cst.parse_module(code)
  for_node = module.body[0]

  transformed = transform_loops(for_node, ctx)
  assert isinstance(transformed, cst.FlattenSentinel)
  assert len(transformed.nodes) == 2

  # Check that Reason is correctly set for not-range
  first_node = transformed.nodes[0]
  comments = [line.comment.value for line in first_node.leading_lines if line.comment]
  assert any("requires structural rewrite" in c for c in comments)


def test_analyze_range_iterator_not_call():
  """Docstring."""
  code = "for i in my_list:\n    pass"
  module = cst.parse_module(code)
  for_node = module.body[0]

  is_range, args = _analyze_range_iterator(for_node.iter)
  assert not is_range
  assert args == []


def test_analyze_range_iterator_not_range_name():
  """Docstring."""
  code = "for i in my_func(10):\n    pass"
  module = cst.parse_module(code)
  for_node = module.body[0]

  is_range, args = _analyze_range_iterator(for_node.iter)
  assert not is_range
  assert args == []
