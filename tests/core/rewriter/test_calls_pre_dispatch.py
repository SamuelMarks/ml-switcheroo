"""Tests for call pre-check integration with ``ApiTransformer`` and dispatch helpers."""

from unittest.mock import patch

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.analysis.symbol_types import SymbolType, TensorType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.enums import LogicOp
from ml_switcheroo.semantics.manager import SemanticsManager


def _make_transformer(source: str = "torch", target: str = "jax") -> ApiTransformer:
  """Build an ``ApiTransformer`` instance backed by a real ``RewriterContext``.

  Args:
      source: Source framework identifier.
      target: Target framework identifier.

  Returns:
      Configured ``ApiTransformer`` instance.
  """
  sem = SemanticsManager()
  cfg = RuntimeConfig(source_framework=source, target_framework=target)
  ctx = RewriterContext(semantics=sem, config=cfg, symbol_table=SymbolTable())
  return ApiTransformer(ctx)


def test_handle_pre_checks_functional_apply_and_stateful() -> None:
  """Verify ``handle_pre_checks`` unwraps functional apply calls and rewrites stateful calls."""
  rewriter = _make_transformer(source="flax_nnx", target="jax")
  rewriter.semantics.framework_configs["flax_nnx"] = {
    "traits": {"functional_execution_method": "apply"},
  }
  rewriter._cached_source_traits = None

  orig = cst.Call(
    func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")),
    args=[cst.Arg(value=cst.Name("vars")), cst.Arg(value=cst.Name("x"))],
  )
  handled, node = handle_pre_checks(rewriter, orig, orig, "layer.apply")
  assert handled is True
  assert isinstance(node, cst.Call)
  assert isinstance(node.func, cst.Name)
  assert node.func.value == "layer"
  assert len(node.args) == 1
  assert cst.ensure_type(node.args[0].value, cst.Name).value == "x"

  rewriter._mark_stateful("cell")
  rewriter.semantics.framework_configs["jax"] = {
    "stateful_call": {"method": "apply", "prepend_arg": "params"},
  }
  state_orig = cst.Call(func=cst.Name("cell"), args=[cst.Arg(value=cst.Name("inp"))])
  handled_state, state_node = handle_pre_checks(rewriter, state_orig, state_orig, "cell")
  assert handled_state is True
  assert isinstance(state_node, cst.Call)
  assert isinstance(state_node.func, cst.Attribute)
  assert cst.ensure_type(state_node.func.value, cst.Name).value == "cell"
  assert state_node.func.attr.value == "apply"


def test_resolve_implicit_method_symbol_table_and_roots() -> None:
  """Verify ``resolve_implicit_method`` resolves symbol table types and implicit roots."""
  rewriter = _make_transformer(source="torch", target="jax")
  rewriter.semantics.framework_configs["torch"] = {
    "traits": {"implicit_method_roots": ["torch.Tensor"]},
  }
  rewriter._cached_source_traits = None

  call_node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("sum")), args=[])
  mappings: dict[str, dict[str, JsonValue]] = {
    "ArrayLike.sum": {"api": "jnp.sum"},
    "torch.Tensor.sum": {"api": "jnp.sum"},
  }

  assert rewriter.context.symbol_table is not None
  with patch.object(rewriter, "_get_mapping", side_effect=lambda name, silent=False: mappings.get(name)):
    with patch.object(rewriter.context.symbol_table, "get_type", return_value=SymbolType(name="ArrayLike")):
      assert resolve_implicit_method(rewriter, call_node, None) == "ArrayLike.sum"

    with patch.object(
      rewriter.context.symbol_table,
      "get_type",
      return_value=TensorType(name="Tensor", framework="torch"),
    ):
      assert resolve_implicit_method(rewriter, call_node, None) == "torch.Tensor.sum"


class MockRule:
  """Typed rule stub used by dispatch condition tests.

  Attributes:
      if_arg: Parameter name inspected by the rule.
      op: Comparison logic operator.
      is_val: Expected value or collection of values.
      use_api: Target API selected when the rule matches.
  """

  def __init__(
    self,
    if_arg: str,
    op: LogicOp | str,
    is_val: JsonValue | set[JsonValue] | tuple[JsonValue, ...] = None,
    use_api: str | None = None,
  ) -> None:
    """Initialize rule fields.

    Args:
        if_arg: Parameter name inspected by the rule.
        op: Comparison logic operator.
        is_val: Expected value or collection of values.
        use_api: Target API selected when the rule matches.
    """
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api
