"""
Tests for Plugin Logic and Hook Execution.

Verifies:
1. Plugin functionality for decomposition.
2. Plugin functionality for recomposition (reverse).
3. Registration mechanics.
"""

import libcst as cst

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import register_hook, get_hook, HookContext
from ml_switcheroo.config import RuntimeConfig


# Helper reused to clean list
def cleanup_args(args_list):
  if args_list:
    args_list[-1] = args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  return args_list


class MockSemantics(SemanticsManager):
  def __init__(self):
    super().__init__()
    # 1. Setup 'special_add' that maps to a plugin on JAX side
    self._reverse_index["torch.special_add"] = (
      "special_add",
      {"variants": {"torch": {"args": {}}, "jax": {"api": "jax.doesnt_matter", "requires_plugin": "mock_alpha_rewrite"}}},
    )
    # 2. Setup standard add for unit testing decompositions directly
    self.data["add"] = {"variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.numpy.add"}}}
    self._reverse_index["jax.numpy.add"] = ("add", self.data["add"])
    self._reverse_index["torch.add"] = ("add", self.data["add"])


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
  engine = ASTEngine(semantics=MockSemantics(), source="torch", target="jax")

  # Input has spaces and commas that LibCST tracks
  code = "y = torch.special_add(x, y, alpha=0.5)"

  result = engine.run(code)

  # Expect clean syntax: plugin_success(x, y)
  assert "plugin_success(x, y)" in result.code
  assert "alpha" not in result.code


def test_real_decomposition_loading():
  """Verify that the real decompositions.py connects."""
  assert get_hook("decompose_alpha") is not None
  assert get_hook("recompose_alpha") is not None


def test_recompose_alpha_logic():
  """
  Unit test for transform_alpha_add_reverse logic.
  Input Node: add(x, y * 5)
  Expected: torch.add(x, y, alpha=5)
  """
  hook = get_hook("recompose_alpha")

  # Create CST Node: jax.numpy.add(x, y * 5)
  # y * 5 is BinaryOperation(left=y, op=Multiply, right=5)

  bin_op = cst.BinaryOperation(left=cst.Name("y"), operator=cst.Multiply(), right=cst.Integer("5"))

  input_node = cst.Call(
    func=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("add")),  # simplified
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Arg(value=bin_op),
    ],
  )

  # Setup Context
  mgr = MockSemantics()
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  ctx = HookContext(mgr, cfg)

  # Execute Hook
  res_node = hook(input_node, ctx)

  # Assert Name Swap
  # _resolve_target_name uses lookup_api -> torch.add
  # cst Name or Attribute structure check tough, convert to code string check?

  # We can inspect the struct
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
  """
  Verify `recompose_alpha` preserves structure if no multiplication exists.
  Input: add(x, y)
  Expected: torch.add(x, y)  (Just name swap)
  """
  hook = get_hook("recompose_alpha")

  input_node = cst.Call(func=cst.Name("add"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])

  mgr = MockSemantics()
  cfg = RuntimeConfig(source_framework="jax", target_framework="torch")
  ctx = HookContext(mgr, cfg)

  res_node = hook(input_node, ctx)

  # Name swapped to torch.add (via default fallback in helper if mock incomplete)
  # but args remain 2
  assert len(res_node.args) == 2
  assert res_node.args[1].keyword is None
