"""Test suite for rewriter calls strategy."""

from ml_switcheroo.enums import RuntimeTensorOrValue


from unittest.mock import Mock, patch

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = Mock()
    self.source_framework = "source_framework"
    self.target_framework = "target_framework"
    self.strict_mode = False
    self._is_module_alias = Mock(return_value=False)
    self.failures: list[RuntimeTensorOrValue] = []
    self.semantics = Mock()
    self.semantics._key_origins = {"some_id": "neural"}

  def _report_failure(self, msg: str) -> None:
    """Execute report failure helper logic.

    Args:
        msg: The msg (str) for report failure.
    """
    self.failures.append(msg)

  def _create_name_node(self, name: str) -> cst.BaseExpression:
    """Execute create name node helper logic.

    Returns:
        Resulting `cst.BaseExpression` from create name node.
    """
    return cst.Name(name)


@pytest.fixture
def original_call() -> cst.Call:
  """Execute original call helper logic.

  Returns:
      Configured `original_call` fixture value (cst.Call).
  """
  return cst.Call(func=cst.Name("foo"), args=[])


@pytest.fixture
def updated_call() -> cst.Call:
  """Execute updated call helper logic.

  Returns:
      Configured `updated_call` fixture value (cst.Call).
  """
  return cst.Call(func=cst.Name("bar"), args=[])


def test_execute_strategy_dispatch_rules(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy dispatch rules behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy dispatch rules.
      updated_call: The updated call (cst.Call) for test execute strategy dispatch rules.
  """
  rewriter = MockRewriter()
  mapping = {"dispatch_rules": ["rule1"]}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = "dispatched_api"
    # We need to test how it handles when the standard strategy is executed after dispatch
    getattr(getattr({"v": mapping}, "__getitem__")("v"), "__setitem__")(
      "api", getattr({"v": "original_api"}, "__getitem__")("v")
    )
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")

    mock_eval.assert_called_once_with(rewriter, original_call, ["rule1"], details)
    assert getattr(getattr(result, "func", None), "value", None) == "dispatched_api"


def test_execute_strategy_variant_imports(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy variant imports behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy variant imports.
      updated_call: The updated call (cst.Call) for test execute strategy variant imports.
  """
  rewriter = MockRewriter()
  setattr(
    getattr({"v": rewriter}, "__getitem__")("v"),
    "_handle_variant_imports",
    getattr({"v": getattr({"v": Mock}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  mapping = {"api": "foo"}
  execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")
  getattr(
    {
      "v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v")._handle_variant_imports}, "__getitem__")(
        "v"
      ).assert_called_once_with
    },
    "__getitem__",
  )("v")(getattr({"v": mapping}, "__getitem__")("v"))


def test_execute_strategy_infix_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy infix success behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy infix success.
      updated_call: The updated call (cst.Call) for test execute strategy infix success.
  """
  rewriter = MockRewriter()
  mapping = {"transformation_type": "infix", "operator": "+"}
  details = {"std_args": ["a", "b"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))]
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.BinaryOperation)


def test_execute_strategy_infix_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy infix failure behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy infix failure.
      updated_call: The updated call (cst.Call) for test execute strategy infix failure.
  """
  rewriter = MockRewriter()
  mapping = {"transformation_type": "infix", "operator": "+"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert getattr({"v": "Infix/Prefix transformation failed:"}, "__getitem__")("v") in getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
    )("v")


def test_execute_strategy_inline_lambda_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy inline lambda success behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy inline lambda success.
      updated_call: The updated call (cst.Call) for test execute strategy inline lambda success.
  """
  rewriter = MockRewriter()
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = []
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.Call)


def test_execute_strategy_inline_lambda_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy inline lambda failure behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy inline lambda failure.
      updated_call: The updated call (cst.Call) for test execute strategy inline lambda failure.
  """
  rewriter = MockRewriter()
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert getattr({"v": "Inline lambda transformation failed:"}, "__getitem__")("v") in getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
    )("v")


def test_execute_strategy_plugin_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy plugin success behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy plugin success.
      updated_call: The updated call (cst.Call) for test execute strategy plugin success.
  """
  rewriter = MockRewriter()
  mapping = {"requires_plugin": "my_plugin"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    expected_node = cst.Name("plugin_result")
    mock_hook = Mock(return_value=expected_node)
    mock_get_hook.return_value = mock_hook
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == expected_node
    mock_hook.assert_called_once_with(updated_call, rewriter.context.hook_context)


def test_execute_strategy_plugin_returns_none(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy plugin returns none behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy plugin returns none.
      updated_call: The updated call (cst.Call) for test execute strategy plugin returns none.
  """
  rewriter = MockRewriter()
  mapping = {"requires_plugin": "my_plugin"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_hook = Mock(return_value=None)
    mock_get_hook.return_value = mock_hook
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    mock_hook.assert_called_once_with(updated_call, rewriter.context.hook_context)


def test_execute_strategy_plugin_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy plugin failure behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy plugin failure.
      updated_call: The updated call (cst.Call) for test execute strategy plugin failure.
  """
  rewriter = MockRewriter()
  mapping = {"requires_plugin": "my_plugin"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_get_hook.return_value = None
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert getattr({"v": "Missing required plugin:"}, "__getitem__")("v") in getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
    )("v")


def test_execute_strategy_macro_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy macro success behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy macro success.
      updated_call: The updated call (cst.Call) for test execute strategy macro success.
  """
  rewriter = MockRewriter()
  mapping = {"macro_template": "{x} + 1"}
  details = {"std_args": ["x", ["y", "def"], {"name": "z"}]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b")), cst.Arg(value=cst.Name("c"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_macro") as mock_macro:
      mock_macro.return_value = "macro_result"
      result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert result == "macro_result"
      mock_macro.assert_called_once_with("{x} + 1", mock_norm.return_value, ["x", "y", "z"])


def test_execute_strategy_macro_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy macro failure behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy macro failure.
      updated_call: The updated call (cst.Call) for test execute strategy macro failure.
  """
  rewriter = MockRewriter()
  mapping = {"macro_template": "{x} + 1"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert getattr({"v": "Macro expansion failed:"}, "__getitem__")("v") in getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
    )("v")


def test_execute_strategy_standard_missing_api(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy standard missing api behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy standard missing api.
      updated_call: The updated call (cst.Call) for test execute strategy standard missing api.
  """
  rewriter = MockRewriter()
  mapping: dict[str, RuntimeTensorOrValue] = {}
  details: dict[str, RuntimeTensorOrValue] = {}

  result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert getattr({"v": "No mapping available"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
  )("v")


def test_execute_strategy_standard_missing_api_neural(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy standard missing api neural behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy standard missing api neural.
      updated_call: The updated call (cst.Call) for test execute strategy standard missing api neural.
  """
  rewriter = MockRewriter()
  rewriter.target_framework = "jax"
  mapping: dict[str, RuntimeTensorOrValue] = {}
  details: dict[str, RuntimeTensorOrValue] = {}

  result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert getattr({"v": "Cannot map neural network abstraction"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
  )("v")


def test_execute_strategy_standard_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy standard success behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy standard success.
      updated_call: The updated call (cst.Call) for test execute strategy standard success.
  """
  rewriter = MockRewriter()
  rewriter.strict_mode = True
  mapping = {"api": "new_api"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards") as mock_guards:
      mock_guards.return_value = [cst.Arg(value=cst.Name("a"))]
      result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert isinstance(result, cst.Call)
      assert getattr(result.func, "value", None) == "new_api"


def test_execute_strategy_standard_layout_map(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy standard layout map behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy standard layout map.
      updated_call: The updated call (cst.Call) for test execute strategy standard layout map.
  """
  rewriter = MockRewriter()
  mapping = {"api": "new_api", "layout_map": {"a": "NCHW->NHWC"}}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy._apply_layout_permutation") as mock_layout:
      mock_layout.return_value = "layout_result"
      result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert result == "layout_result"


def test_execute_strategy_standard_normalization_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy standard normalization failure behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy standard normalization failure.
      updated_call: The updated call (cst.Call) for test execute strategy standard normalization failure.
  """
  rewriter = MockRewriter()
  mapping = {"api": "new_api"}
  details: dict[str, RuntimeTensorOrValue] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert getattr({"v": "Argument normalization failed"}, "__getitem__")("v") in getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").failures}, "__getitem__")("v")[0]}, "__getitem__"
    )("v")


def test_apply_layout_permutation() -> None:
  """Verify apply layout permutation behavior."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {
    "layout_map": {
      "a": "NCHW->NHWC",
      "return": "NHWC->NCHW",
    }
  }
  details = {"std_args": ["a", ["b", "c"], {"name": "d"}]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.side_effect = [(0, 2, 3, 1), (0, 3, 1, 2)]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inject:
      mock_inject.side_effect = [cst.Name("permuted_x"), cst.Name("permuted_return")]

      result = _apply_layout_permutation(node, mapping, details, rewriter)

      assert mock_compute.call_count == 2
      assert mock_inject.call_count == 2
      assert getattr(result, "value", None) == "permuted_return"


def test_execute_strategy_no_hook_context(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy no hook context behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy no hook context.
      updated_call: The updated call (cst.Call) for test execute strategy no hook context.
  """
  rewriter = MockRewriter()
  del rewriter.context.hook_context
  mapping = {"api": "foo"}
  _res = execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")
  assert isinstance(_res, cst.Call)
  assert getattr(_res.func, "value", None) == "foo"


def test_execute_strategy_dispatch_rules_falsy(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Verify execute strategy dispatch rules falsy behavior.

  Args:
      original_call: The original call (cst.Call) for test execute strategy dispatch rules falsy.
      updated_call: The updated call (cst.Call) for test execute strategy dispatch rules falsy.
  """
  rewriter = MockRewriter()
  mapping = {"dispatch_rules": ["rule1"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = None
    res = execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")
  assert res is updated_call


def test_apply_layout_permutation_no_arrow() -> None:
  """Verify apply layout permutation no arrow behavior."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {
    "layout_map": {
      "a": "NCHW",
      "return": "NHWC",
    }
  }
  details = {"std_args": ["a"]}
  result = _apply_layout_permutation(node, mapping, details, rewriter)
  assert result.deep_equals(node)


def test_apply_layout_permutation_falsy_indices() -> None:
  """Verify apply layout permutation falsy indices behavior."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {
    "layout_map": {
      "a": "NCHW->NHWC",
      "return": "NHWC->NCHW",
    }
  }
  details = {"std_args": ["a"]}
  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.return_value = None
    result = _apply_layout_permutation(node, mapping, details, rewriter)
    assert result.deep_equals(node)


def test_apply_layout_permutation_out_of_bounds_idx() -> None:
  """Verify apply layout permutation out of bounds idx behavior."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[])
  mapping = {
    "layout_map": {
      "a": "NCHW->NHWC",
    }
  }
  details = {"std_args": ["a"]}
  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.return_value = (0, 2, 3, 1)
    result = _apply_layout_permutation(node, mapping, details, rewriter)
    assert getattr(
      {"v": len(getattr({"v": getattr({"v": result}, "__getitem__")("v").args}, "__getitem__")("v"))}, "__getitem__"
    )("v") == getattr({"v": 0}, "__getitem__")("v")
