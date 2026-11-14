"""Unit tests for call strategy execution (`ml_switcheroo.core.rewriter.calls.strategy`)."""

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import Rule
from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.core.hooks_registry import clear_hooks
from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.enums import LogicOp, SemanticTier
from ml_switcheroo.semantics.manager import SemanticsManager


def _parse_call(code: str) -> cst.Call:
  """Parse a Python call expression into `cst.Call`.

  Args:
      code: Python call expression string.

  Returns:
      Parsed `cst.Call` node.
  """
  return cst.ensure_type(cst.parse_expression(code), cst.Call)


def _expr_code(node: cst.CSTNode) -> str:
  """Serialize a rewritten CST expression to stripped Python source code.

  Args:
      node: Rewritten CST expression node.

  Returns:
      Formatted Python expression string.
  """
  assert isinstance(node, cst.BaseExpression)
  return cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code.strip()


def _make_transformer(*, strict_mode: bool = False) -> ApiTransformer:
  """Construct a real `ApiTransformer` backed by `SemanticsManager` and `RewriterContext`.

  Args:
      strict_mode: Whether to enable strict argument/rank guard wrapping.

  Returns:
      Configured `ApiTransformer` instance.
  """
  sem = SemanticsManager()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=strict_mode)
  ctx = RewriterContext(semantics=sem, config=cfg)
  return ApiTransformer(ctx)


def test_execute_strategy_standard_dispatch_and_strict_guards() -> None:
  """Verify standard API rewriting, conditional `dispatch_rules`, `required_imports`, and `strict_mode` guards."""
  transformer = _make_transformer(strict_mode=False)
  call = _parse_call("torch.add(x, y)")
  details: dict[str, JsonValue] = {"std_args": ["a", "b"]}
  mapping: dict[str, JsonValue] = {
    "api": "jnp.add",
    "required_imports": [{"module": "jax.numpy", "alias": "jnp"}],
  }

  res = execute_strategy(transformer, call, call, mapping, details, "Add")
  assert _expr_code(res) == "jnp.add(x, y)"
  assert "import jax.numpy as jnp" in transformer.context.module_preamble

  # Conditional dispatch rule overrides `api` when matched, and falls back to `api` when unmatched
  rules = [Rule(if_arg="mode", op=LogicOp.EQ, is_val="nearest", use_api="jax.image.resize_nearest")]
  dispatch_details: dict[str, JsonValue] = {"std_args": ["x", "mode"]}
  matched = execute_strategy(
    transformer,
    _parse_call("interp(img, 'nearest')"),
    _parse_call("interp(img, 'nearest')"),
    {"api": "jax.image.resize", "dispatch_rules": rules},
    dispatch_details,
    "Interp",
  )
  assert _expr_code(matched) == "jax.image.resize_nearest(img, 'nearest')"

  unmatched = execute_strategy(
    transformer,
    _parse_call("interp(img, 'cubic')"),
    _parse_call("interp(img, 'cubic')"),
    {"api": "jax.image.resize", "dispatch_rules": rules},
    dispatch_details,
    "Interp",
  )
  assert _expr_code(unmatched) == "jax.image.resize(img, 'cubic')"

  # Strict mode injects rank assertion guard helper when `rank` constraint is specified
  strict_transformer = _make_transformer(strict_mode=True)
  strict_details: dict[str, JsonValue] = {"std_args": [{"name": "x", "rank": 4}]}
  strict_res = execute_strategy(
    strict_transformer,
    _parse_call("conv(x=img)"),
    _parse_call("conv(x=img)"),
    {"api": "lax.conv"},
    strict_details,
    "Conv",
  )
  assert _expr_code(strict_res) == "lax.conv(x=_check_rank(img, 4))"


def test_execute_strategy_infix_inline_lambda_and_macro() -> None:
  """Verify real `infix`, `inline_lambda`, and `macro_template` strategies and error fallbacks."""
  transformer = _make_transformer()

  # Infix binary operator and invalid operator fallback
  infix_call = _parse_call("torch.add(x, y)")
  infix_res = execute_strategy(
    transformer,
    infix_call,
    infix_call,
    {"transformation_type": "infix", "operator": "+"},
    {"std_args": ["a", "b"]},
    "Add",
  )
  assert _expr_code(infix_res) == "x + y"

  transformer.context.current_stmt_errors.clear()
  bad_infix = execute_strategy(
    transformer,
    infix_call,
    infix_call,
    {"transformation_type": "infix", "operator": "???"},
    {"std_args": ["a", "b"]},
    "Add",
  )
  assert bad_infix is infix_call
  assert any("Infix/Prefix transformation failed" in e for e in transformer.context.current_stmt_errors)

  # Inline lambda and syntax-error fallback
  lam_call = _parse_call("relu6(val)")
  lam_res = execute_strategy(
    transformer,
    lam_call,
    lam_call,
    {"transformation_type": "inline_lambda", "api": "lambda x: jnp.minimum(jnp.maximum(x, 0), 6)"},
    {"std_args": ["x"]},
    "Relu6",
  )
  assert _expr_code(lam_res) == "(lambda x: jnp.minimum(jnp.maximum(x, 0), 6))(val)"

  transformer.context.current_stmt_errors.clear()
  bad_lam = execute_strategy(
    transformer,
    lam_call,
    lam_call,
    {"transformation_type": "inline_lambda", "api": "lambda: ????"},
    {"std_args": ["x"]},
    "Relu6",
  )
  assert bad_lam is lam_call
  assert any("Inline lambda transformation failed" in e for e in transformer.context.current_stmt_errors)

  # Macro template expansion (supporting str, list/tuple, and dict `std_args`) and failure fallback
  macro_call = _parse_call("add_scaled(u, v, 2)")
  macro_res = execute_strategy(
    transformer,
    macro_call,
    macro_call,
    {"macro_template": "{a} + {b} * {scale}"},
    {"std_args": ["a", ["b", "Tensor"], {"name": "scale"}]},
    "AddScaled",
  )
  assert _expr_code(macro_res) == "u + v * 2"

  transformer.context.current_stmt_errors.clear()
  bad_macro = execute_strategy(
    transformer,
    _parse_call("add_scaled()"),
    _parse_call("add_scaled()"),
    {"macro_template": "{missing_arg} + 1"},
    {"std_args": ["missing_arg"]},
    "AddScaled",
  )
  assert _expr_code(bad_macro) == "add_scaled()"
  assert any("Macro expansion failed" in e for e in transformer.context.current_stmt_errors)


def test_execute_strategy_plugins_missing_mappings_and_normalization_errors() -> None:
  """Verify plugin hooks, missing plugins, neural vs non-neural missing mappings, and norm failures."""
  transformer = _make_transformer()
  call = _parse_call("custom_op(x)")

  clear_hooks(mark_loaded=True)

  @register_hook("test_custom_double")
  def _double_hook(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    """Rewrite `custom_op(x)` into `x * 2`.

    Args:
        node: Input `cst.Call` node.
        ctx: Active `HookContext`.

    Returns:
        Binary multiplication CST node.
    """
    assert ctx.current_op_id == "CustomOp"
    return cst.BinaryOperation(left=node.args[0].value, operator=cst.Multiply(), right=cst.Integer("2"))

  @register_hook("test_noop_none")
  def _none_hook(_node: cst.Call, _ctx: HookContext) -> None:
    """Return `None` so `execute_strategy` preserves `updated_node`.

    Args:
        _node: Input `cst.Call` node.
        _ctx: Active `HookContext`.
    """

  assert (
    _expr_code(execute_strategy(transformer, call, call, {"requires_plugin": "test_custom_double"}, {}, "CustomOp"))
    == "x * 2"
  )
  assert execute_strategy(transformer, call, call, {"requires_plugin": "test_noop_none"}, {}, "CustomOp") is call

  transformer.context.current_stmt_errors.clear()
  assert execute_strategy(transformer, call, call, {"requires_plugin": "nonexistent_hook"}, {}, "CustomOp") is call
  assert any("Missing required plugin" in e for e in transformer.context.current_stmt_errors)

  clear_hooks(mark_loaded=False)

  # Missing API: custom message, default message, and neural tier message
  transformer.context.current_stmt_errors.clear()
  execute_strategy(transformer, call, call, {"missing_message": "Custom unsupported reason"}, {}, "CustomOp")
  assert transformer.context.current_stmt_errors == ["Custom unsupported reason"]

  transformer.context.current_stmt_errors.clear()
  execute_strategy(transformer, call, call, {}, {}, "CustomOp")
  assert "No mapping available for 'CustomOp' -> 'jax'" in transformer.context.current_stmt_errors[0]

  transformer.semantics._key_origins["NeuralLayer"] = SemanticTier.NEURAL.value
  transformer.context.current_stmt_errors.clear()
  execute_strategy(transformer, call, call, {}, {}, "NeuralLayer")
  assert "Cannot map neural network abstraction" in transformer.context.current_stmt_errors[0]

  # Standard normalization / guard ValueError failure
  strict_err_transformer = _make_transformer(strict_mode=True)
  res_bad_norm = execute_strategy(
    strict_err_transformer,
    _parse_call("foo(a=x)"),
    _parse_call("foo(a=x)"),
    {"api": "jnp.foo"},
    {"std_args": [{"name": "a", "rank": "not_an_int"}]},
    "Foo",
  )
  assert _expr_code(res_bad_norm) == "foo(a=x)"
  assert any("Argument normalization failed" in e for e in strict_err_transformer.context.current_stmt_errors)

  # Host without `context.hook_context` raises PluginExecutionError on plugin, handles missing `_key_origins`, and executes standard calls
  import pytest

  from ml_switcheroo.exceptions import PluginExecutionError

  class _ContextWithoutHook:
    """Minimal context lacking `hook_context`."""

  class _NoOriginsSemantics:
    """Semantics stub without `_key_origins`."""

  class _MinimalHost:
    """Minimal strategy host lacking `hook_context`, `_handle_variant_imports`, and `semantics`."""

    context = _ContextWithoutHook()
    source_framework = "torch"
    target_framework = "jax"
    strict_mode = False

    def _is_module_alias(self, _node: cst.CSTNode) -> bool:
      """Return `False` for alias check.

      Args:
          _node: Candidate CST node.

      Returns:
          Always `False`.
      """
      return False

    def _create_name_node(self, name: str) -> cst.BaseExpression:
      """Construct `cst.Name` from `name`.

      Args:
          name: Function name string.

      Returns:
          `cst.Name` node.
      """
      return cst.Name(name)

    def _report_failure(self, _msg: str) -> None:
      """No-op failure reporter.

      Args:
          _msg: Failure message.
      """

  class _NoOriginsHost(_MinimalHost):
    """Host exposing `semantics` without `_key_origins`."""

    semantics = _NoOriginsSemantics()

  min_res = execute_strategy(
    _MinimalHost(), _parse_call("f(x)"), _parse_call("f(x)"), {"api": "g"}, {"std_args": ["x"]}, "F"
  )
  assert _expr_code(min_res) == "g(x)"
  assert _apply_layout_permutation(
    _parse_call("f(x)"), {"layout_map": {"x": "NCHW->NHWC"}}, {"std_args": ["x"]}, _MinimalHost()
  ).deep_equals(_parse_call("f(x)"))
  execute_strategy(_NoOriginsHost(), _parse_call("f(x)"), _parse_call("f(x)"), {}, {}, "F")

  clear_hooks(mark_loaded=True)

  @register_hook("test_needs_ctx")
  def _needs_ctx(node: cst.Call, _ctx: HookContext) -> cst.CSTNode:
    """No-op hook requiring a non-None `HookContext`.

    Args:
        node: Input call node.
        _ctx: Hook context.

    Returns:
        Unchanged call node.
    """
    return node

  with pytest.raises(PluginExecutionError, match="requires a valid HookContext"):
    execute_strategy(
      _MinimalHost(), _parse_call("f(x)"), _parse_call("f(x)"), {"requires_plugin": "test_needs_ctx"}, {}, "F"
    )
  clear_hooks(mark_loaded=False)


def test_layout_permutation_rewriting_and_edge_cases() -> None:
  """Verify real NCHW<->NHWC argument and return permutation injection as well as invalid/out-of-bounds rules."""
  transformer = _make_transformer()
  conv_call = _parse_call("conv(x, w)")
  mapping: dict[str, JsonValue] = {
    "api": "lax_conv",
    "layout_map": {
      "input": "NCHW->NHWC",
      "return": "NHWC->NCHW",
      "missing_param": "NCHW->NHWC",
    },
  }
  details: dict[str, JsonValue] = {"std_args": [{"name": "input"}, ["weight", "Tensor"], "bias"]}

  rewritten = execute_strategy(transformer, conv_call, conv_call, mapping, details, "Conv2d")
  code = _expr_code(rewritten)
  assert "lax_conv(" in code
  assert "(0, 2, 3, 1)" in code
  assert "(0, 3, 1, 2)" in code

  # Invalid layout rules (no `->`, mismatched rank, or out-of-bounds positional argument) leave call intact
  no_arrow = _apply_layout_permutation(
    _parse_call("f(x)"),
    {"layout_map": {"x": "NCHW", "return": "NHWC"}},
    {"std_args": ["x"]},
    transformer,
  )
  assert _expr_code(no_arrow) == "f(x)"

  bad_perm = _apply_layout_permutation(
    _parse_call("f(x)"),
    {"layout_map": {"x": "NCHW->NHW", "return": "NHW->NCHW"}},
    {"std_args": ["x"]},
    transformer,
  )
  assert _expr_code(bad_perm) == "f(x)"

  oob = _apply_layout_permutation(
    _parse_call("f()"),
    {"layout_map": {"x": "NCHW->NHWC"}},
    {"std_args": ["x"]},
    transformer,
  )
  assert _expr_code(oob) == "f()"
