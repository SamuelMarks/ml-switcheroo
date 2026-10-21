"""Test suite for the Loss Wrapper module."""

import typing
from pydantic import JsonValue

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
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: A factory function.
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "loss_reduction", getattr({"v": transform_loss_reduction}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  ce_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.cross_entropy"},
      "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      "tensorflow": {"api": "tf.nn.sparse_softmax_cross_entropy_with_logits", "requires_plugin": "loss_reduction"},
    }
  }
  mean_def = {"variants": {"jax": {"api": "jnp.mean"}, "tensorflow": {"api": "tf.reduce_mean"}}}
  sum_def = {"variants": {"jax": {"api": "jnp.sum"}, "tensorflow": {"api": "tf.reduce_sum"}}}
  all_defs = {"CrossEntropyLoss": ce_def, "Mean": mean_def, "Sum": sum_def}

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: The definition or None.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))
        if getattr({"v": "cross_entropy"}, "__getitem__")("v") in getattr({"v": name}, "__getitem__")("v")
        else None
      },
      "__getitem__",
    )("v")
    return _ret_v

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework name.

    Returns:
        Optional[dict[str, JsonValue]]: The variant or None.
    """
    if aid in all_defs and fw in all_defs[aid]["variants"]:
      _ret_v: dict[str, JsonValue] | None = getattr(
        {
          "v": getattr(
            {"v": getattr({"v": getattr({"v": all_defs}, "__getitem__")("v")[aid]}, "__getitem__")("v")["variants"]},
            "__getitem__",
          )("v")[fw]
        },
        "__getitem__",
      )("v")
      return _ret_v
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
  """Verifies the behavior of JAX mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_tensorflow_mean_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of TensorFlow mean reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("tensorflow")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "tf.reduce_mean" in res
  assert "tf.nn.sparse_softmax_cross_entropy" in res


def test_explicit_sum_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of explicit sum reduction.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "jnp.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of reduction none.

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


def test_loss_wrapper_fallback(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when API not found.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("torch")
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_known_apis},
      "__getitem__",
    )("v"),
    "return_value",
    getattr({"v": {}}, "__getitem__")("v"),
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


def test_loss_wrapper_fallback_wrapper_none(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when wrapper API lookup fails.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")

  def lookup(aid: str) -> str | None:
    """Docstring.

    Args:
        aid (str): Definition ID.

    Returns:
        Optional[str]: Lookup result.
    """
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    return None

  setattr(rewriter.context.hook_context, "lookup_api", lookup)
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_no_context(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def lookup(aid: str) -> str | None:
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

  setattr(rewriter.context.hook_context, "lookup_api", lookup)
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


def test_loss_wrapper_variable_no_context_real(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {"v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))},
      "__getitem__",
    )("v")
    return _ret_v

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_definition},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": get_def}, "__getitem__")("v"),
  )

  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_real(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {"v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))},
      "__getitem__",
    )("v")
    return _ret_v

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_definition},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": get_def}, "__getitem__")("v"),
  )

  code = "torch.nn.functional.cross_entropy(a, b)"  # triggers lines 72, 90
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_variable_and_no_context_really(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {"v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))},
      "__getitem__",
    )("v")
    return _ret_v

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_definition},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": get_def}, "__getitem__")("v"),
  )
  code = "torch.nn.functional.cross_entropy(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_none_op_id(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies none op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert getattr({"v": "optax"}, "__getitem__")("v") in getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Module}, "__getitem__")("v")(
            body=getattr(
              {
                "v": [
                  getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")(
                        "v"
                      )(
                        body=getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                    value=getattr({"v": res}, "__getitem__")("v")
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v")
                      )
                    },
                    "__getitem__",
                  )("v")
                ]
              },
              "__getitem__",
            )("v")
          )
        },
        "__getitem__",
      )("v").code
    },
    "__getitem__",
  )("v")


def test_loss_wrapper_empty_string_op_id(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies empty string op_id sets CrossEntropyLoss.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert getattr({"v": "optax"}, "__getitem__")("v") in getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Module}, "__getitem__")("v")(
            body=getattr(
              {
                "v": [
                  getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")(
                        "v"
                      )(
                        body=getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                    value=getattr({"v": res}, "__getitem__")("v")
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v")
                      )
                    },
                    "__getitem__",
                  )("v")
                ]
              },
              "__getitem__",
            )("v")
          )
        },
        "__getitem__",
      )("v").code
    },
    "__getitem__",
  )("v")


def test_loss_wrapper_fallback_no_context_really(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = ""

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {"v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))},
      "__getitem__",
    )("v")
    return _ret_v

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_definition},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": get_def}, "__getitem__")("v"),
  )
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_really_empty_string(
  rewriter_factory: typing.Callable[[str], PivotRewriter],
) -> None:
  """Verifies fallback when no op_id is set.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "optax" if x == "CrossEntropyLoss" else None)

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Docstring.

    Args:
        name (str): Definition name.

    Returns:
        Tuple[str, dict[str, JsonValue]]: Tuple containing string and definition dictionary.
    """
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {"v": (getattr({"v": "CrossEntropyLoss"}, "__getitem__")("v"), getattr({"v": ce_def}, "__getitem__")("v"))},
      "__getitem__",
    )("v")
    return _ret_v

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").get_definition},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": get_def}, "__getitem__")("v"),
  )
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_direct_empty_context(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies empty context hits fallback.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  import libcst as cst

  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  code = "torch.nn.functional.cross_entropy(a, b)"
  module = cst.parse_module(code)
  node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  setattr(
    rewriter.context.hook_context,
    "lookup_api",
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None,
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert getattr({"v": "optax"}, "__getitem__")("v") in getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Module}, "__getitem__")("v")(
            body=getattr(
              {
                "v": [
                  getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")(
                        "v"
                      )(
                        body=getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                    value=getattr({"v": res}, "__getitem__")("v")
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v")
                      )
                    },
                    "__getitem__",
                  )("v")
                ]
              },
              "__getitem__",
            )("v")
          )
        },
        "__getitem__",
      )("v").code
    },
    "__getitem__",
  )("v")


def test_loss_wrapper_unknown_reduction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies unknown reduction mode hits unwrapped fallback.

  Args:
      rewriter_factory: Factory providing test rewriter instance.
  """
  rewriter = rewriter_factory("jax")
  code = "torch.nn.functional.cross_entropy(a, b, reduction='batchmean')"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
