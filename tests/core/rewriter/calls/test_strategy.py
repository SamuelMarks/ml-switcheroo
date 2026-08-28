"""Test suite for strategy.py"""

import libcst as cst
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy, _apply_layout_permutation


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  expr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr)


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.hook_context = MagicMock()
    self.hook_context.current_op_id = None


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context = MockContext()
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.strict_mode = False
    self._report_failure = MagicMock()
    self.semantics = MagicMock()
    self._handle_variant_imports = MagicMock()

  def _create_name_node(self, api_str: str) -> cst.Name:
    """Docstring."""
    return cst.Name(api_str)

  def _is_module_alias(self, node: typing.Any) -> bool:
    """Docstring."""
    return False


@patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules", return_value="dispatched_api")
@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", return_value=[cst.Arg(value=cst.Name("x"))])
def test_execute_strategy_dispatch_and_standard(mock_norm: MagicMock, mock_eval: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping: dict[str, typing.Any] = {"dispatch_rules": ["rule"]}
  details: dict[str, typing.Any] = {}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, details, "id")  # type: ignore
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, result.func).value == "dispatched_api"
  rewriter._handle_variant_imports.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_infix", return_value=cst.Name("infix_res"))
def test_execute_strategy_infix(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("add(x, y)")
  updated = parse_call("add(x, y)")
  mapping: dict[str, typing.Any] = {"transformation_type": "infix", "operator": "+"}
  details: dict[str, typing.Any] = {}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, details, "add")  # type: ignore
  assert isinstance(result, cst.Name)
  assert result.value == "infix_res"


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("fail"))
def test_execute_strategy_infix_fail(mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("add(x, y)")
  updated = parse_call("add(x, y)")
  mapping: dict[str, typing.Any] = {"transformation_type": "infix"}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "add")  # type: ignore
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_inline_lambda", return_value=cst.Name("lambda_res"))
def test_execute_strategy_inline_lambda(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"transformation_type": "inline_lambda", "api": "lambda x: x"}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result.value == "lambda_res"


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError)
def test_execute_strategy_inline_lambda_fail(mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"transformation_type": "inline_lambda", "api": "lambda x: x"}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"requires_plugin": "my_plugin"}

  mock_hook = MagicMock(return_value=cst.Name("plugin_res"))
  mock_get_hook.return_value = mock_hook

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result.value == "plugin_res"

  mock_get_hook.return_value = None
  result_fail: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result_fail == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_macro", return_value=cst.Name("macro_res"))
def test_execute_strategy_macro(mock_rewrite: MagicMock, mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"macro_template": "mac"}
  details: dict[str, typing.Any] = {"std_args": ["a", ["b", "int"], {"name": "c"}]}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, details, "id")  # type: ignore
  assert result.value == "macro_res"
  mock_rewrite.assert_called_once()
  assert mock_rewrite.call_args[0][2] == ["a", "b", "c"]


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=Exception)
def test_execute_strategy_macro_fail(mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"macro_template": "mac"}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result == updated
  rewriter._report_failure.assert_called_once()


def test_execute_strategy_standard_missing_api() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments")
@patch("ml_switcheroo.core.rewriter.calls.strategy._apply_layout_permutation", return_value=parse_call("permuted()"))
@patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards", return_value=[cst.Arg(value=cst.Name("y"))])
def test_execute_strategy_standard_success(mock_guards: MagicMock, mock_permute: MagicMock, mock_norm: MagicMock) -> None:
  """Docstring."""
  mock_norm.return_value = [cst.Arg(value=cst.Name("x"))]
  rewriter = MockRewriter()
  rewriter.strict_mode = True

  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping: dict[str, typing.Any] = {"api": "new_func", "layout_map": {}}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert typing.cast(cst.Name, result.func).value == "new_func"
  mock_guards.assert_called_once()


@patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError)
def test_execute_strategy_standard_norm_fail(mock_norm: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func()")
  updated = parse_call("func()")
  mapping: dict[str, typing.Any] = {"api": "new_func"}

  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert result == updated
  rewriter._report_failure.assert_called_once()


@patch(
  "ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call", side_effect=lambda v, p, s, t: cst.Name("permuted")
)
@patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation", return_value=(0, 2, 1))
def test_apply_layout_permutation(mock_compute: MagicMock, mock_inject: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func(a, b)")
  mapping: dict[str, typing.Any] = {"layout_map": {"a": "NCHW->NHWC", "return": "NHWC->NCHW", "not_found": "A->B"}}
  details: dict[str, typing.Any] = {"std_args": [{"name": "a"}, ["b", "int"]]}

  result: typing.Any = _apply_layout_permutation(node, mapping, details, rewriter)  # type: ignore
  assert typing.cast(cst.Name, result).value == "permuted"

  assert mock_compute.call_count == 2
  assert mock_inject.call_count == 2


@patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation", return_value=None)
def test_apply_layout_permutation_invalid(mock_compute: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func(a)")
  mapping: dict[str, typing.Any] = {"layout_map": {"a": "NCHW->NHWC", "return": "NHWC->NCHW"}}
  details: dict[str, typing.Any] = {"std_args": ["a"]}

  result: typing.Any = _apply_layout_permutation(node, mapping, details, rewriter)  # type: ignore
  # Original node because permutations are invalid
  assert typing.cast(cst.Name, result.func).value == "func"
  assert typing.cast(cst.Name, result.args[0].value).value == "a"


def test_execute_strategy_standard_success_layout() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("func(x)")
  updated = parse_call("func(x)")
  mapping: dict[str, typing.Any] = {"api": "new_func", "layout_map": {"return": "A->B"}}
  result: typing.Any = execute_strategy(rewriter, original, updated, mapping, {}, "id")  # type: ignore
  assert typing.cast(cst.Name, result.func).value == "new_func"  # permute fails, returns original node
