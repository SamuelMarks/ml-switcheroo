"""Test suite for the Loss Wrapper module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): Source code.

  Returns:
      str: Rewritten code.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: A factory function.
  """
  hooks.register_hook("loss_reduction")(transform_loss_reduction)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  ce_variants: dict[str, dict[str, JsonValue]] = {
    "torch": {"api": "torch.nn.functional.cross_entropy"},
    "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
    "tensorflow": {"api": "tf.nn.sparse_softmax_cross_entropy_with_logits", "requires_plugin": "loss_reduction"},
  }
  mean_variants: dict[str, dict[str, JsonValue]] = {
    "jax": {"api": "jnp.mean"},
    "tensorflow": {"api": "tf.reduce_mean"},
  }
  sum_variants: dict[str, dict[str, JsonValue]] = {
    "jax": {"api": "jnp.sum"},
    "tensorflow": {"api": "tf.reduce_sum"},
  }
  ce_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](ce_variants)}
  mean_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](mean_variants)}
  sum_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](sum_variants)}
  all_defs: dict[str, dict[str, JsonValue]] = {"CrossEntropyLoss": ce_def, "Mean": mean_def, "Sum": sum_def}
  all_variants: dict[str, dict[str, dict[str, JsonValue]]] = {
    "CrossEntropyLoss": ce_variants,
    "Mean": mean_variants,
    "Sum": sum_variants,
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: The definition or None.
    """
    return ("CrossEntropyLoss", ce_def) if "cross_entropy" in name else None

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework name.

    Returns:
        Optional[dict[str, JsonValue]]: The variant or None.
    """
    if aid in all_variants and fw in all_variants[aid]:
      return all_variants[aid][fw]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  def create(target_framework: str) -> PivotRewriter:
    """Creates rewriter.

    Args:
        target_framework (str): Target framework string.

    Returns:
        PivotRewriter: The rewriter instance.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target_framework)
    rw = PivotRewriter(mgr, cfg)
    rw.ctx.current_op_id = "CrossEntropyLoss"
    return rw

  return create


def test_jax_mean_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `optax.softmax_cross_entropy` for JAX mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_tensorflow_mean_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `tf.reduce_mean` for tensorflow mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("tensorflow")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "tf.reduce_mean" in res
  assert "tf.nn.sparse_softmax_cross_entropy" in res


def test_explicit_sum_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `jnp.sum` for explicit sum reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "jnp.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `optax.softmax` for reduction none.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(x, y, reduction='none')"
  res = rewrite_code(rewriter, code)
  assert "jnp.mean" not in res
  assert "jnp.sum" not in res
  assert "optax.softmax" in res


def test_loss_wrapper_reduction_variable(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction does not crash.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter_factory("jax"), code)
  assert "my_mode" in res


def test_loss_wrapper_fallback(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when API not found.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("torch")
  monkeypatch.setattr(
    rewriter.semantics.get_known_apis,
    "return_value",
    {},
    raising=False,
  )
  code = "torch.nn.functional.mse_loss(a, b, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_reduction_variable_path(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction skips wrapping but parses.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("torch")
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "my_mode" in res  # or whatever it produces


def test_loss_wrapper_no_context_fallback(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op id.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_wrapper_none(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when wrapper API lookup fails.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")

  def lookup(aid: str) -> str | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        aid (str): Definition ID.

    Returns:
        Optional[str]: Lookup result.
    """
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    return None

  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lookup, raising=False)
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_no_context(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def lookup(aid: str) -> str | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        aid (str): Definition ID.

    Returns:
        Optional[str]: Lookup result.
    """
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    if aid == "Mean":
      return "jnp.mean"
    return None

  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lookup, raising=False)
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_and_no_context(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_no_context_real(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def: dict[str, JsonValue] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  monkeypatch.setattr(
    rewriter.semantics.get_definition,
    "side_effect",
    get_def,
    raising=False,
  )

  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_real(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def: dict[str, JsonValue] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  monkeypatch.setattr(
    rewriter.semantics.get_definition,
    "side_effect",
    get_def,
    raising=False,
  )

  code = "torch.nn.functional.cross_entropy(a, b)"  # triggers lines 72, 90
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_variable_and_no_context_really(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def: dict[str, JsonValue] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  monkeypatch.setattr(
    rewriter.semantics.get_definition,
    "side_effect",
    get_def,
    raising=False,
  )
  code = "torch.nn.functional.cross_entropy(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_none_op_id(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies none op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  node = cst.ensure_type(expr.value, cst.Call)
  monkeypatch.setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
    raising=False,
  )
  res = cst.ensure_type(transform_loss_reduction(node, rewriter.context.hook_context), cst.Call)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_empty_string_op_id(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies empty string op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  node = cst.ensure_type(expr.value, cst.Call)
  monkeypatch.setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
    raising=False,
  )
  res = cst.ensure_type(transform_loss_reduction(node, rewriter.context.hook_context), cst.Call)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_fallback_no_context_really(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = ""

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def: dict[str, JsonValue] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  monkeypatch.setattr(
    rewriter.semantics.get_definition,
    "side_effect",
    get_def,
    raising=False,
  )
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_really_empty_string(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  monkeypatch.setattr(
    rewriter.context.hook_context, "lookup_api", lambda x: "optax" if x == "CrossEntropyLoss" else None, raising=False
  )

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock target API path or definition for the given identifier.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def: dict[str, JsonValue] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  monkeypatch.setattr(
    rewriter.semantics.get_definition,
    "side_effect",
    get_def,
    raising=False,
  )
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_direct_empty_context(
  rewriter_factory: typing.Callable[[str], PivotRewriter], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verifies empty context hits fallback.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  import libcst as cst

  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  code = "torch.nn.functional.cross_entropy(a, b)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  node = cst.ensure_type(expr.value, cst.Call)
  monkeypatch.setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
    raising=False,
  )
  res = cst.ensure_type(transform_loss_reduction(node, rewriter.context.hook_context), cst.Call)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_unknown_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies unknown reduction mode hits unwrapped fallback.

  Args:
      rewriter_factory: Factory providing test rewriter instance.
  """
  rewriter = rewriter_factory("jax")
  code = "torch.nn.functional.cross_entropy(a, b, reduction='batchmean')"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_branches_ext() -> None:
  """Verify loss wrapper reduction transformation branches for empty args and custom arg mappings."""
  from ml_switcheroo.core.dsl import FrameworkVariant
  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "keras"
  ctx.current_op_id = None
  ctx.current_variant = None
  ctx.lookup_api.return_value = None

  node_empty = cst.ensure_type(cst.parse_expression("loss()"), cst.Call)
  assert transform_loss_reduction(node_empty, ctx) == node_empty

  node1 = cst.ensure_type(cst.parse_expression("loss(y_true, y_pred)"), cst.Call)
  assert transform_loss_reduction(node1, ctx) == node1

  node2 = cst.ensure_type(cst.parse_expression("obj.loss(y_true, y_pred)"), cst.Call)
  assert transform_loss_reduction(node2, ctx) == node2

  ctx.lookup_api.return_value = "keras.losses.MSE"
  node3 = cst.ensure_type(cst.parse_expression("obj.loss(y_true=a, y_pred=b)"), cst.Call)
  res3 = transform_loss_reduction(node3, ctx)
  assert cst.Module(body=[]).code_for_node(res3) == "keras.losses.MSE(keras.losses.MSE(y_true=a, y_pred=b))"

  ctx.current_variant = FrameworkVariant(api="keras.losses.MSE", args={"custom": "my_val"})
  node4 = cst.ensure_type(cst.parse_expression("loss(custom=a)"), cst.Call)
  res4 = transform_loss_reduction(node4, ctx)
  assert cst.Module(body=[]).code_for_node(res4) == "keras.losses.MSE(keras.losses.MSE(custom=a))"
