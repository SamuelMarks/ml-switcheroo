"""Tests for call pre-processing and implicit method resolution against ``RewriterHost``."""

import typing
from unittest.mock import patch

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.analysis.symbol_types import SymbolType, TensorType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.manager import SemanticsManager


class _NoOpBackend(CompilerBackend):
  """Minimal concrete ``CompilerBackend`` used to verify default ``configure_emission``."""

  def compile(self, graph: LogicalGraph) -> str:
    """Return an empty artifact string.

    Args:
        graph: Input logical graph.

    Returns:
        Empty string.
    """
    return ""


def _parse_call(expr_code: str) -> cst.Call:
  """Parse a Python call expression string into a ``cst.Call`` node.

  Args:
      expr_code: Source string representing a single call expression.

  Returns:
      Parsed ``cst.Call`` node.
  """
  expr = cst.parse_expression(expr_code)
  assert isinstance(expr, cst.Call)
  return expr


def _make_rewriter(
  *,
  source: str = "torch",
  target: str = "jax",
  symbol_table: SymbolTable | None = None,
) -> ApiTransformer:
  """Construct a real ``ApiTransformer`` bound to a ``RewriterContext`` and ``SemanticsManager``.

  Args:
      source: Source framework identifier.
      target: Target framework identifier.
      symbol_table: Optional pre-populated ``SymbolTable``.

  Returns:
      Initialized ``ApiTransformer`` conforming to ``RewriterHost``.
  """
  sem = SemanticsManager()
  cfg = RuntimeConfig(source_framework=source, target_framework=target)
  ctx = RewriterContext(semantics=sem, config=cfg, symbol_table=symbol_table)
  return ApiTransformer(ctx)


@pytest.fixture(autouse=True)
def _cleanup_hooks() -> typing.Generator[None, None, None]:
  """Isolate global hook registrations across each test in this module.

  Yields:
      None while the test executes.
  """
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks(mark_loaded=True)
  yield
  hr.clear_hooks()


def test_backend_configure_emission_and_target_traits() -> None:
  """Verify ``CompilerBackend.configure_emission`` and ``ApiTransformer.target_traits``."""
  noop = _NoOpBackend()
  noop.configure_emission(emit_trivia_comments=True)
  assert noop.semantics is None

  sass = NvidiaSassBackend()
  sass.configure_emission(emit_trivia_comments=True)
  assert bool(sass.synthesizer.emit_trivia_comments) is True
  sass2 = NvidiaSassBackend()
  sass2.configure_emission(emit_trivia_comments=True)
  sass2.configure_emission(emit_trivia_comments=False)
  assert bool(sass2.synthesizer.emit_trivia_comments) is False

  rdna = RdnaBackend()
  rdna.configure_emission(emit_trivia_comments=True)
  assert bool(rdna.synthesizer.emit_trivia_comments) is True
  rdna2 = RdnaBackend()
  rdna2.configure_emission(emit_trivia_comments=True)
  rdna2.configure_emission(emit_trivia_comments=False)
  assert bool(rdna2.synthesizer.emit_trivia_comments) is False

  rewriter = _make_rewriter(source="torch", target="jax")
  assert rewriter.target_traits == rewriter._get_target_traits()


def test_handle_pre_checks_functional_unwrap() -> None:
  """Verify functional ``.apply(vars, x)`` unwrapping and non-matching fallthroughs."""
  rewriter = _make_rewriter(source="flax_nnx", target="torch")
  rewriter.semantics.framework_configs["flax_nnx"] = {
    "traits": {"functional_execution_method": "apply"},
  }
  rewriter._cached_source_traits = None

  call_with_args = _parse_call("layer.apply(variables, x)")
  handled, res = handle_pre_checks(rewriter, call_with_args, call_with_args, "layer.apply")
  assert handled is True
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Name)
  assert res.func.value == "layer"
  assert len(res.args) == 1

  call_no_args = _parse_call("layer.apply()")
  handled_empty, res_empty = handle_pre_checks(rewriter, call_no_args, call_no_args, "layer.apply")
  assert handled_empty is True
  assert isinstance(res_empty, cst.Call)
  assert len(res_empty.args) == 0

  # Updated call already rewritten to non-Attribute -> falls through
  non_attr_updated = _parse_call("f()")
  handled_non_attr, _ = handle_pre_checks(rewriter, call_no_args, non_attr_updated, "layer.apply")
  assert handled_non_attr is False


def test_handle_pre_checks_plugin_claim_and_inplace_unrolling() -> None:
  """Verify plugin claims bypass heuristic unrolling while explicit/heuristic in-place ops unroll."""
  rewriter = _make_rewriter(source="torch", target="jax")
  node = _parse_call("foo()")

  assert handle_pre_checks(rewriter, node, node, None) == (False, node)
  assert handle_pre_checks(rewriter, node, node, "unknown_fn") == (False, node)

  # Explicit plugin claim skips heuristic unrolling and lifecycle stripping
  with patch.object(rewriter, "_get_mapping", return_value={"requires_plugin": "my_plugin"}):
    assert handle_pre_checks(rewriter, _parse_call("x.add_()"), _parse_call("x.add_()"), "x.add_")[0] is False

  # Non-plugin mapping and non-inplace definition fall through
  with (
    patch.object(rewriter, "_get_mapping", return_value={"api": "jnp.add"}),
    patch.object(rewriter.semantics, "get_definition", return_value=("Add", {"is_inplace": False})),
  ):
    assert handle_pre_checks(rewriter, node, node, "torch.add") == (False, node)

  # Definition with is_inplace=True triggers unroll_inplace_ops hook when registered
  with (
    patch.object(rewriter, "_get_mapping", return_value=None),
    patch.object(rewriter.semantics, "get_definition", return_value=("AddInplace", {"is_inplace": True})),
  ):
    # No hook registered yet
    assert handle_pre_checks(rewriter, node, node, "inplace_fn")[0] is False

    # Hook returns unchanged node
    @register_hook("unroll_inplace_ops")
    def _noop_hook(n: cst.Call, ctx: HookContext) -> cst.CSTNode:
      """Return input node unchanged.

      Args:
          n: Input CST Call node.
          ctx: Active HookContext.

      Returns:
          Unchanged CST node.
      """
      return n

    assert handle_pre_checks(rewriter, node, node, "inplace_fn")[0] is False

    clear_hooks(mark_loaded=True)

    # Hook returns non-expression statement -> ignored safely
    @register_hook("unroll_inplace_ops")
    def _stmt_hook(n: cst.Call, ctx: HookContext) -> cst.CSTNode:
      """Return a non-expression CST node.

      Args:
          n: Input CST Call node.
          ctx: Active HookContext.

      Returns:
          Empty CST Pass statement.
      """
      return cst.Pass()

    assert handle_pre_checks(rewriter, node, node, "inplace_fn")[0] is False

    clear_hooks(mark_loaded=True)

    # Hook returns transformed expression -> handled=True
    @register_hook("unroll_inplace_ops")
    def _replace_hook(n: cst.Call, ctx: HookContext) -> cst.CSTNode:
      """Replace in-place call with functional expression.

      Args:
          n: Input CST Call node.
          ctx: Active HookContext.

      Returns:
          Replacement CST Name expression.
      """
      return cst.Name("functional_add")

    handled, replaced = handle_pre_checks(rewriter, node, node, "inplace_fn")
    assert handled is True
    assert isinstance(replaced, cst.Name)
    assert replaced.value == "functional_add"

  # Trailing-underscore heuristic (excluding dunder names) also triggers unrolling
  with (
    patch.object(rewriter, "_get_mapping", return_value=None),
    patch.object(rewriter.semantics, "get_definition", return_value=None),
  ):
    dunder_call = _parse_call("__init__()")
    assert handle_pre_checks(rewriter, dunder_call, dunder_call, "__init__")[0] is False
    underscore_call = _parse_call("foo_()")
    handled_u, replaced_u = handle_pre_checks(rewriter, underscore_call, underscore_call, "foo_")
    assert handled_u is True
    assert isinstance(replaced_u, cst.Name)


def test_handle_pre_checks_lifecycle_and_stateful_calls() -> None:
  """Verify lifecycle strip/warn method handling and stateful call rewriting."""
  rewriter = _make_rewriter(source="torch", target="jax")
  rewriter.semantics.framework_configs["torch"] = {
    "traits": {
      "lifecycle_strip_methods": ["cuda", "detach"],
      "lifecycle_warn_methods": ["eval"],
    }
  }
  rewriter._cached_source_traits = None

  strip_call = _parse_call("tensor.cuda()")
  handled_s, stripped = handle_pre_checks(rewriter, strip_call, strip_call, "tensor.cuda")
  assert handled_s is True
  assert isinstance(stripped, cst.Name)
  assert stripped.value == "tensor"
  assert any("cuda" in w for w in rewriter.context.current_stmt_warnings)

  # Non-attribute updated call falls through strip/warn
  non_attr = _parse_call("fn()")
  assert handle_pre_checks(rewriter, strip_call, non_attr, "tensor.cuda")[0] is False

  warn_call = _parse_call("model.eval()")
  handled_w, warned = handle_pre_checks(rewriter, warn_call, warn_call, "model.eval")
  assert handled_w is True
  assert isinstance(warned, cst.Name)
  assert warned.value == "model"
  assert any("eval" in w for w in rewriter.context.current_stmt_warnings)
  assert handle_pre_checks(rewriter, warn_call, non_attr, "model.eval")[0] is False

  # Non-lifecycle attribute method falls through
  other_attr = _parse_call("model.forward(x)")
  assert handle_pre_checks(rewriter, other_attr, other_attr, "model.forward")[0] is False

  # Stateful call rewriting when target config specifies stateful_call
  rewriter._mark_stateful("layer")
  state_call = _parse_call("layer(x)")
  rewriter.semantics.framework_configs["jax"] = {}
  assert handle_pre_checks(rewriter, state_call, state_call, "layer")[0] is False

  rewriter.semantics.framework_configs["jax"] = {
    "stateful_call": {"method": "apply", "prepend_arg": "variables"},
  }
  handled_st, rewritten_st = handle_pre_checks(rewriter, state_call, state_call, "layer")
  assert handled_st is True
  assert isinstance(rewriter_mod := cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(rewritten_st)])]), cst.Module)
  assert rewriter_mod.code.strip() == "layer.apply(variables, x)"


def test_resolve_implicit_method_all_branches() -> None:
  """Verify ``resolve_implicit_method`` across self, module aliases, symbol table, and implicit roots."""
  sym_table = SymbolTable()
  rewriter = _make_rewriter(source="torch", target="jax", symbol_table=sym_table)
  rewriter.semantics.framework_configs["torch"] = {
    "traits": {"implicit_method_roots": ["torch.Tensor"]},
  }
  rewriter._cached_source_traits = None
  rewriter.context.alias_map[" alias_mod "] = "torch"
  rewriter.context.alias_map["alias_mod"] = "torch"

  # Non-attribute call -> None
  assert resolve_implicit_method(rewriter, _parse_call("fn()"), "fn") is None

  # self.method() -> None
  assert resolve_implicit_method(rewriter, _parse_call("self.reshape(2, 2)"), "self.reshape") is None

  # Module alias receiver -> None
  assert resolve_implicit_method(rewriter, _parse_call("alias_mod.reshape(x)"), "alias_mod.reshape") is None

  mappings: dict[str, dict[str, JsonValue]] = {
    "CustomType.custom_op": {"api": "jnp.custom_op"},
    "torch.Tensor.view": {"api": "jnp.reshape"},
  }

  with patch.object(rewriter, "_get_mapping", side_effect=lambda name, silent=False: mappings.get(name)):
    # 1. SymbolTable non-Tensor symbol hit
    x_call = _parse_call("x.custom_op()")
    assert isinstance(x_call.func, cst.Attribute)
    with patch.object(sym_table, "get_type", return_value=SymbolType(name="CustomType")):
      assert resolve_implicit_method(rewriter, x_call, "x.custom_op") == "CustomType.custom_op"

    # SymbolTable hit where mapping is missing falls back to implicit_method_roots
    y_view = _parse_call("y.view(4)")
    with patch.object(sym_table, "get_type", return_value=SymbolType(name="UnmappedType")):
      assert resolve_implicit_method(rewriter, y_view, "y.view") == "torch.Tensor.view"

    # SymbolTable TensorType with framework prefix
    with patch.object(sym_table, "get_type", return_value=TensorType(name="Tensor", framework="torch")):
      assert resolve_implicit_method(rewriter, y_view, "y.view") == "torch.Tensor.view"

    # SymbolTable TensorType with empty framework falls back to Tensor.<method>
    with patch.object(sym_table, "get_type", return_value=TensorType(name="Tensor", framework="")):
      assert resolve_implicit_method(rewriter, _parse_call("y.unmapped()"), "y.unmapped") is None

    # SymbolTable returns None -> implicit roots fallback
    with patch.object(sym_table, "get_type", return_value=None):
      assert resolve_implicit_method(rewriter, y_view, "y.view") == "torch.Tensor.view"

    # Rewriter with symbol_table = None -> implicit roots fallback
    rewriter_no_sym = _make_rewriter(source="torch", target="jax", symbol_table=None)
    rewriter_no_sym.semantics.framework_configs["torch"] = {
      "traits": {"implicit_method_roots": ["torch.Tensor"]},
    }
    rewriter_no_sym._cached_source_traits = None
    with patch.object(rewriter_no_sym, "_get_mapping", side_effect=lambda name, silent=False: mappings.get(name)):
      assert resolve_implicit_method(rewriter_no_sym, y_view, "y.view") == "torch.Tensor.view"
      assert resolve_implicit_method(rewriter_no_sym, _parse_call("y.missing()"), "y.missing") is None
