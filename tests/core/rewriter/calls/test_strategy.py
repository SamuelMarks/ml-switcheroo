"""Test suite for strategy.py"""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy


def parse_call(code: str) -> cst.Call:
  """Execute parse call helper logic.

  Args:
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  module = cst.parse_module(code)
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call)


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context = MagicMock()
    self.hook_context.current_op_id = None


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = MockContext()
    self.source_framework = "torch"
    self.target_framework = "jax"
    self.strict_mode = False
    self._report_failure = MagicMock()
    self.semantics = MagicMock()
    self._handle_variant_imports = MagicMock()

  def _create_name_node(self, api_str: str) -> cst.Name:
    """Execute create name node helper logic.

    Args:
        api_str: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(api_str)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Execute is module alias helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return False


@patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules", return_value="dispatched_api")
@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", return_value=[cst.Arg(value=cst.Name("x"))])
def test_execute_strategy_dispatch_and_standard(mock_norm: MagicMock, mock_eval: MagicMock) -> None:
  """Verify execute strategy dispatch and standard behavior.

  Args:
      mock_norm: Test argument or fixture.
      mock_eval: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping = {"dispatch_rules": ["rule"]}
  details: dict[str, typing.Hashable | typing.Sized] = {}

  result = execute_strategy(rewriter, original, updated, mapping, details, "id")
  assert isinstance(result, cst.Call)
  assert cst.ensure_type(result.func, cst.Name).value == "dispatched_api"
  rewriter._handle_variant_imports.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_infix", return_value=cst.Name("infix_res"))
def test_execute_strategy_infix(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Verify execute strategy infix behavior.

  Args:
      mock_rewrite: Test argument or fixture.
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("add(x, y)")
  updated = parse_call("add(x, y)")
  mapping = {"transformation_type": "infix", "operator": "+"}
  details: dict[str, typing.Hashable | typing.Sized] = {}

  result = execute_strategy(rewriter, original, updated, mapping, details, "add")
  assert isinstance(result, cst.Name)
  assert result.value == "infix_res"


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("fail"))
def test_execute_strategy_infix_fail(mock_norm: MagicMock) -> None:
  """Verify execute strategy infix fail behavior.

  Args:
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("add(x, y)")
  updated = parse_call("add(x, y)")
  mapping = {"transformation_type": "infix"}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "add")
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_inline_lambda", return_value=cst.Name("lambda_res"))
def test_execute_strategy_inline_lambda(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Verify execute strategy inline lambda behavior.

  Args:
      mock_rewrite: Test argument or fixture.
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert getattr({"v": getattr({"v": result}, "__getitem__")("v").value}, "__getitem__")("v") == getattr(
    {"v": "lambda_res"}, "__getitem__"
  )("v")


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError)
def test_execute_strategy_inline_lambda_fail(mock_norm: MagicMock) -> None:
  """Verify execute strategy inline lambda fail behavior.

  Args:
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook: MagicMock) -> None:
  """Verify execute strategy plugin behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"requires_plugin": "my_plugin"}

  mock_hook = MagicMock(return_value=cst.Name("plugin_res"))
  mock_get_hook.return_value = mock_hook

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert getattr({"v": getattr({"v": result}, "__getitem__")("v").value}, "__getitem__")("v") == getattr(
    {"v": "plugin_res"}, "__getitem__"
  )("v")

  mock_get_hook.return_value = None
  result_fail = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert result_fail == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_macro", return_value=cst.Name("macro_res"))
def test_execute_strategy_macro(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Verify execute strategy macro behavior.

  Args:
      mock_rewrite: Test argument or fixture.
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"macro_template": "mac"}
  details = {"std_args": ["a", ["b", "int"], {"name": "c"}]}

  result = execute_strategy(rewriter, original, updated, mapping, details, "id")
  assert getattr({"v": getattr({"v": result}, "__getitem__")("v").value}, "__getitem__")("v") == getattr(
    {"v": "macro_res"}, "__getitem__"
  )("v")
  mock_rewrite.assert_called_once()
  assert mock_rewrite.call_args[0][2] == ["a", "b", "c"]


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=Exception)
def test_execute_strategy_macro_fail(mock_norm: MagicMock) -> None:
  """Verify execute strategy macro fail behavior.

  Args:
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"macro_template": "mac"}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert result == updated
  rewriter._report_failure.assert_called_once()


def test_execute_strategy_standard_missing_api() -> None:
  """Verify execute strategy standard missing api behavior."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Hashable | typing.Sized] = {}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy._apply_layout_permutation", return_value=parse_call("permuted()"))
@patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards", return_value=[cst.Arg(value=cst.Name("y"))])
def test_execute_strategy_standard_success(mock_guards: MagicMock, mock_permute: MagicMock, mock_norm: MagicMock) -> None:
  """Verify execute strategy standard success behavior.

  Args:
      mock_guards: Test argument or fixture.
      mock_permute: Test argument or fixture.
      mock_norm: Test argument or fixture.
  """
  mock_norm.return_value = [cst.Arg(value=cst.Name("x"))]
  rewriter = MockRewriter()
  rewriter.strict_mode = True

  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping = {"api": "new_func", "layout_map": {}}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "new_func"}, "__getitem__")("v")
  mock_guards.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError)
def test_execute_strategy_standard_norm_fail(mock_norm: MagicMock) -> None:
  """Verify execute strategy standard norm fail behavior.

  Args:
      mock_norm: Test argument or fixture.
  """
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping = {"api": "new_func"}

  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch(
  "ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call", side_effect=lambda v, p, s, t: cst.Name("permuted")
)
@patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation", return_value=(0, 2, 1))
def test_apply_layout_permutation(mock_compute: MagicMock, mock_inject: MagicMock) -> None:
  """Verify apply layout permutation behavior.

  Args:
      mock_compute: Test argument or fixture.
      mock_inject: Test argument or fixture.
  """
  rewriter = MockRewriter()
  node = parse_call("func(a, b)")
  mapping = {"layout_map": {"a": "NCHW->NHWC", "return": "NHWC->NCHW", "not_found": "A->B"}}
  details = {"std_args": [{"name": "a"}, ["b", "int"]]}

  result = _apply_layout_permutation(node, mapping, details, rewriter)
  assert cst.ensure_type(result, cst.Name).value == "permuted"

  assert mock_compute.call_count == 2
  assert mock_inject.call_count == 2


@patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation", return_value=None)
def test_apply_layout_permutation_invalid(mock_compute: MagicMock) -> None:
  """Verify apply layout permutation invalid behavior.

  Args:
      mock_compute: Test argument or fixture.
  """
  rewriter = MockRewriter()
  node = parse_call("func(a)")
  mapping = {"layout_map": {"a": "NCHW->NHWC", "return": "NHWC->NCHW"}}
  details = {"std_args": ["a"]}

  result = _apply_layout_permutation(node, mapping, details, rewriter)
  # Original node because permutations are invalid
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "func"}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").args}, "__getitem__")("v")[0]},
                  "__getitem__",
                )("v").value
              },
              "__getitem__",
            )("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "a"}, "__getitem__")("v")


def test_execute_strategy_standard_success_layout() -> None:
  """Verify execute strategy standard success layout behavior."""
  rewriter = MockRewriter()
  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping = {"api": "new_func", "layout_map": {"return": "A->B"}}
  result = execute_strategy(rewriter, original, updated, mapping, {}, "id")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "new_func"}, "__getitem__")("v")
