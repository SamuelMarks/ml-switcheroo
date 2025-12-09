"""
Tests for Plugin Logic and Hook Execution.
"""

import libcst as cst
import pytest
from unittest.mock import MagicMock
from typing import Set

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import register_hook, get_hook, HookContext, _HOOKS, clear_hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins import decompositions  # ensure load


# Helper reused to clean list
def cleanup_args(args_list):
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  def __init__(self):
    # Do NOT call super().__init__(), it loads real files and resets data
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}
    self.framework_configs = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # 1. 'special_add'
    special_def = {
      "variants": {
        "torch": {"api": "torch.special_add", "args": {}},
        "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"},
      },
      "std_args": ["x", "y"],
    }
    self.data["special_add"] = special_def
    self._reverse_index["torch.special_add"] = ("special_add", special_def)

    # 2. 'add' (Standard)
    add_def = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self.data["add"] = add_def
    self._reverse_index["jax.numpy.add"] = ("add", add_def)
    self._reverse_index["torch.add"] = ("add", add_def)

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id, target_fw):
    if abstract_id in self.data:
      return self.data[abstract_id]["variants"].get(target_fw)
    return None

  def is_verified(self, _id):
    return True


@register_hook("mock_alpha_rewrite")
def mock_plugin_logic(node, _ctx):
  """
  Rewrite to 'plugin_success(x, y)' removing alpha.
  """
  new_func = cst.Name("plugin_success")
  # Filter alpha
  filtered = [a for a in node.args if not (a.keyword and a.keyword.value == "alpha")]
  filtered = cleanup_args(filtered)
  return node.with_changes(func=new_func, args=filtered)


def test_plugin_trigger():
  # Force registration in case previous tests cleared it
  _HOOKS["mock_alpha_rewrite"] = mock_plugin_logic

  # Use our FIXED mock semantics
  mgr = MockSemantics()
  # Check integrity before run
  assert mgr.get_definition("torch.special_add") is not None

  engine = ASTEngine(semantics=mgr, source="torch", target="jax")

  # Input has spaces and commas that LibCST tracks
  code = "y = torch.special_add(x, y, alpha=0.5)"

  result = engine.run(code)

  # Expect clean syntax: plugin_success(x, y)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_real_decomposition_loading():
  """Verify that the real decompositions.py connects."""
  # Ensure module imported
  assert decompositions.transform_alpha_add is not None
  # Check registry
  assert "decompose_alpha" in _HOOKS


def test_recompose_alpha_logic():
  """
  Unit test for transform_alpha_add_reverse logic.
  """
  from ml_switcheroo.plugins.decompositions import transform_alpha_add_reverse as hook

  # Create CST Node: jax.numpy.add(x, y * 5)
  # y * 5 is BinaryOperation(left=y, op=Multiply, right=5)
  bin_op = cst.BinaryOperation(left=cst.Name("y"), operator=cst.Multiply(), right=cst.Integer("5"))

  input_node = cst.Call(
    func=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("add")),
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=bin_op),
    ],
  )

  # Setup Context
  mgr = MockSemantics()
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  # Must manually populate ctx helper for lookup_api
  ctx = HookContext(mgr, cfg)

  # Mock lookup_api used by _resolve_target_name in plugin
  # _resolve_target_name calls ctx.lookup_api("add") -> should return "torch.add"
  ctx.lookup_api = MagicMock(return_value="torch.add")

  # Execute Hook
  res_node = hook(input_node, ctx)

  # Expected args: x, y, alpha=5
  assert len(res_node.args) == 3

  arg0, arg1, arg2 = res_node.args

  assert isinstance(arg0.value, cst.Name) and arg0.value.value == "x"
  assert isinstance(arg1.value, cst.Name) and arg1.value.value == "y"

  # Arg 2 should be Keyword argument alpha=5
  assert arg2.keyword is not None
  assert arg2.keyword.value == "alpha"
  assert isinstance(arg2.value, cst.Integer) and arg2.value.value == "5"


def test_recompose_alpha_ignores_simple_calls():
  from ml_switcheroo.plugins.decompositions import transform_alpha_add_reverse as hook

  input_node = cst.Call(func=cst.Name("add"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])

  mgr = MockSemantics()
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  ctx = HookContext(mgr, cfg)

  ctx.lookup_api = lambda op: "torch.add" if op == "add" else None

  res_node = hook(input_node, ctx)

  # Name swapped to torch.add but args remain 2
  assert len(res_node.args) == 2
  assert res_node.args[1].keyword is None
