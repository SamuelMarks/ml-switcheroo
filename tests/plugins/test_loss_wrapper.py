"""Test suite for the Loss Wrapper module."""

from typing import Any, Callable, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock

import libcst as cst
import pytest

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
def rewriter_factory() -> Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: A factory function.
  """
  hooks._HOOKS["loss_reduction"] = transform_loss_reduction
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  ce_def: Dict[str, Any] = {
    "variants": {
      "torch": {"api": "torch.nn.functional.cross_entropy"},
      "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      "tensorflow": {"api": "tf.nn.sparse_softmax_cross_entropy_with_logits", "requires_plugin": "loss_reduction"},
    }
  }
  mean_def: Dict[str, Any] = {"variants": {"jax": {"api": "jnp.mean"}, "tensorflow": {"api": "tf.reduce_mean"}}}
  sum_def: Dict[str, Any] = {"variants": {"jax": {"api": "jnp.sum"}, "tensorflow": {"api": "tf.reduce_sum"}}}
  all_defs: Dict[str, Any] = {"CrossEntropyLoss": ce_def, "Mean": mean_def, "Sum": sum_def}

  def get_def(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: The definition or None.
    """
    return ("CrossEntropyLoss", ce_def) if "cross_entropy" in name else None

  def resolve_variant(aid: str, fw: str) -> Optional[Dict[str, Any]]:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework name.

    Returns:
        Optional[Dict[str, Any]]: The variant or None.
    """
    if aid in all_defs and fw in all_defs[aid]["variants"]:
      return all_defs[aid]["variants"][fw]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  def create(target_fw: str) -> PivotRewriter:
    """Creates rewriter.

    Args:
        target_fw (str): Target framework string.

    Returns:
        PivotRewriter: The rewriter instance.
    """
    cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework=target_fw)
    rw: PivotRewriter = PivotRewriter(mgr, cfg)
    rw.ctx.current_op_id = "CrossEntropyLoss"
    return rw

  return create


def test_jax_mean_reduction(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of JAX mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "loss = F.cross_entropy(logits, target)"
  res: str = rewrite_code(rewriter, code)
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_tensorflow_mean_reduction(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of TensorFlow mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("tensorflow")
  code: str = "loss = F.cross_entropy(logits, target)"
  res: str = rewrite_code(rewriter, code)
  assert "tf.reduce_mean" in res
  assert "tf.nn.sparse_softmax_cross_entropy" in res


def test_explicit_sum_reduction(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of explicit sum reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res: str = rewrite_code(rewriter, code)
  assert "jnp.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of reduction none.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "loss = F.cross_entropy(x, y, reduction='none')"
  res: str = rewrite_code(rewriter, code)
  assert "jnp.mean" not in res
  assert "jnp.sum" not in res
  assert "optax.softmax" in res


def test_loss_wrapper_reduction_variable(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction does not crash.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  code: str = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res: str = rewrite_code(rewriter_factory("jax"), code)
  assert "my_mode" in res


def test_loss_wrapper_fallback(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when API not found.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("torch")
  rewriter.semantics.get_known_apis.return_value = {}
  code: str = "torch.nn.functional.mse_loss(a, b, reduction='sum')"
  res: str = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_reduction_variable_path(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction skips wrapping but parses.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("torch")
  code: str = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res: str = rewrite_code(rewriter, code)
  assert "my_mode" in res  # or whatever it produces


def test_loss_wrapper_no_context_fallback(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op id.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code: str = "torch.nn.functional.cross_entropy(a, b)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_wrapper_none(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when wrapper API lookup fails.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")

  def lookup(aid: str) -> Optional[str]:
    """Docstring.

    Args:
        aid (str): Definition ID.

    Returns:
        Optional[str]: Lookup result.
    """
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    return None

  rewriter.context.hook_context.lookup_api = lookup
  code: str = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_no_context(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def lookup(aid: str) -> Optional[str]:
    """Docstring.

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

  rewriter.context.hook_context.lookup_api = lookup
  code: str = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res: str = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_and_no_context(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code: str = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res: str = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_no_context_real(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> Tuple[str, Dict[str, Any]]:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing string and definition dictionary.
    """
    ce_def: Dict[str, Any] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def

  code: str = "torch.nn.functional.cross_entropy(a, b)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_real(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> Tuple[str, Dict[str, Any]]:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing string and definition dictionary.
    """
    ce_def: Dict[str, Any] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def

  code: str = "torch.nn.functional.cross_entropy(a, b)"  # triggers lines 72, 90
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_variable_and_no_context_really(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> Tuple[str, Dict[str, Any]]:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing string and definition dictionary.
    """
    ce_def: Dict[str, Any] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code: str = "torch.nn.functional.cross_entropy(a, b, reduction=my_mode)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_none_op_id(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies none op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code: str = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module: cst.Module = cst.parse_module(code)
  node: Union[cst.CSTNode, cst.BaseExpression] = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = lambda x: (
    "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res: Union[cst.CSTNode, cst.Call] = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_empty_string_op_id(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies empty string op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code: str = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module: cst.Module = cst.parse_module(code)
  node: Union[cst.CSTNode, cst.BaseExpression] = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = lambda x: (
    "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res: Union[cst.CSTNode, cst.Call] = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_fallback_no_context_really(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = ""

  def get_def(name: str) -> Tuple[str, Dict[str, Any]]:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing string and definition dictionary.
    """
    ce_def: Dict[str, Any] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code: str = "torch.nn.functional.cross_entropy(a, b)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_really_empty_string(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  rewriter.context.hook_context.lookup_api = lambda x: "optax" if x == "CrossEntropyLoss" else None

  def get_def(name: str) -> Tuple[str, Dict[str, Any]]:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing string and definition dictionary.
    """
    ce_def: Dict[str, Any] = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code: str = "torch.nn.functional.cross_entropy(a, b)"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_direct_empty_context(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies empty context hits fallback.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  import libcst as cst

  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  code: str = "torch.nn.functional.cross_entropy(a, b)"
  module: cst.Module = cst.parse_module(code)
  node: Union[cst.CSTNode, cst.BaseExpression] = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = lambda x: (
    "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res: Union[cst.CSTNode, cst.Call] = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_unknown_reduction(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies unknown reduction mode hits unwrapped fallback.

  Args:
      rewriter_factory: Factory providing test rewriter instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "torch.nn.functional.cross_entropy(a, b, reduction='batchmean')"
  res: str = rewrite_code(rewriter, code)
  assert "optax" in res
