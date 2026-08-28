"""Test module."""

import pytest
import libcst as cst
from unittest.mock import Mock, patch

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy, _apply_layout_permutation
from typing import Dict, Any, List


class MockRewriter:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.context: Mock = Mock()
    self.source_fw: str = "source_fw"
    self.target_fw: str = "target_fw"
    self.strict_mode: bool = False
    self._is_module_alias: Mock = Mock(return_value=False)
    self.failures: List[str] = []
    self.semantics: Mock = Mock()
    self.semantics._key_origins = {"some_id": "neural"}

  def _report_failure(self, msg: str) -> None:
    self.failures.append(msg)

  def _create_name_node(self, name: str) -> cst.Name:
    return cst.Name(name)


@pytest.fixture
def original_call() -> cst.Call:
  """Test element."""
  return cst.Call(func=cst.Name("foo"), args=[])


@pytest.fixture
def updated_call() -> cst.Call:
  """Test element."""
  return cst.Call(func=cst.Name("bar"), args=[])


def test_execute_strategy_dispatch_rules(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"dispatch_rules": ["rule1"]}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = "dispatched_api"
    # We need to test how it handles when the standard strategy is executed after dispatch
    mapping["api"] = "original_api"
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")

    mock_eval.assert_called_once_with(rewriter, original_call, ["rule1"], details)
    assert getattr(getattr(result, "func", None), "value", None) == "dispatched_api"


def test_execute_strategy_variant_imports(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  rewriter._handle_variant_imports = Mock()
  mapping: Dict[str, Any] = {"api": "foo"}
  execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")
  rewriter._handle_variant_imports.assert_called_once_with(mapping)


def test_execute_strategy_infix_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"transformation_type": "infix", "operator": "+"}
  details: Dict[str, Any] = {"std_args": ["a", "b"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))]
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.BinaryOperation)


def test_execute_strategy_infix_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"transformation_type": "infix", "operator": "+"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Infix/Prefix transformation failed:" in rewriter.failures[0]


def test_execute_strategy_inline_lambda_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = []
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.Call)


def test_execute_strategy_inline_lambda_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Inline lambda transformation failed:" in rewriter.failures[0]


def test_execute_strategy_plugin_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"requires_plugin": "my_plugin"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_hook: Mock = Mock(return_value="plugin_result")
    mock_get_hook.return_value = mock_hook
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == "plugin_result"
    mock_hook.assert_called_once_with(updated_call, rewriter.context.hook_context)


def test_execute_strategy_plugin_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"requires_plugin": "my_plugin"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_get_hook.return_value = None
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Missing required plugin:" in rewriter.failures[0]


def test_execute_strategy_macro_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"macro_template": "{x} + 1"}
  details: Dict[str, Any] = {"std_args": ["x", ["y", "def"], {"name": "z"}]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b")), cst.Arg(value=cst.Name("c"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.rewrite_as_macro") as mock_macro:
      mock_macro.return_value = "macro_result"
      result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert result == "macro_result"
      mock_macro.assert_called_once_with("{x} + 1", mock_norm.return_value, ["x", "y", "z"])


def test_execute_strategy_macro_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"macro_template": "{x} + 1"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Macro expansion failed:" in rewriter.failures[0]


def test_execute_strategy_standard_missing_api(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {}
  details: Dict[str, Any] = {}

  result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert "No mapping available" in rewriter.failures[0]


def test_execute_strategy_standard_missing_api_neural(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  rewriter.target_fw = "jax"
  mapping: Dict[str, Any] = {}
  details: Dict[str, Any] = {}

  result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert "Cannot map neural network abstraction" in rewriter.failures[0]


def test_execute_strategy_standard_success(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  rewriter.strict_mode = True
  mapping: Dict[str, Any] = {"api": "new_api"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards") as mock_guards:
      mock_guards.return_value = [cst.Arg(value=cst.Name("a"))]
      result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert isinstance(result, cst.Call)
      assert getattr(result.func, "value", None) == "new_api"


def test_execute_strategy_standard_layout_map(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"api": "new_api", "layout_map": {"a": "NCHW->NHWC"}}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy._apply_layout_permutation") as mock_layout:
      mock_layout.return_value = "layout_result"
      result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert result == "layout_result"


def test_execute_strategy_standard_normalization_failure(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"api": "new_api"}
  details: Dict[str, Any] = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result: cst.CSTNode = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Argument normalization failed" in rewriter.failures[0]


def test_apply_layout_permutation() -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: Dict[str, Any] = {
    "layout_map": {
      "a": "NCHW->NHWC",
      "return": "NHWC->NCHW",
    }
  }
  details: Dict[str, Any] = {"std_args": ["a", ["b", "c"], {"name": "d"}]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.side_effect = [(0, 2, 3, 1), (0, 3, 1, 2)]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inject:
      mock_inject.side_effect = [cst.Name("permuted_x"), cst.Name("permuted_return")]

      result: cst.Call = _apply_layout_permutation(node, mapping, details, rewriter)

      assert mock_compute.call_count == 2
      assert mock_inject.call_count == 2
      assert getattr(result, "value", None) == "permuted_return"


def test_execute_strategy_no_hook_context(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  del rewriter.context.hook_context
  mapping: Dict[str, Any] = {"api": "foo"}
  execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")


def test_execute_strategy_dispatch_rules_falsy(original_call: cst.Call, updated_call: cst.Call) -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  mapping: Dict[str, Any] = {"dispatch_rules": ["rule1"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = None
    execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")


def test_apply_layout_permutation_no_arrow() -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: Dict[str, Any] = {
    "layout_map": {
      "a": "NCHW",
      "return": "NHWC",
    }
  }
  details: Dict[str, Any] = {"std_args": ["a"]}
  result: cst.CSTNode = _apply_layout_permutation(node, mapping, details, rewriter)
  assert result.deep_equals(node)


def test_apply_layout_permutation_falsy_indices() -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: Dict[str, Any] = {
    "layout_map": {
      "a": "NCHW->NHWC",
      "return": "NHWC->NCHW",
    }
  }
  details: Dict[str, Any] = {"std_args": ["a"]}
  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.return_value = None
    result: cst.CSTNode = _apply_layout_permutation(node, mapping, details, rewriter)
    assert result.deep_equals(node)


def test_apply_layout_permutation_out_of_bounds_idx() -> None:
  """Test element."""
  rewriter: MockRewriter = MockRewriter()
  node: cst.Call = cst.Call(func=cst.Name("foo"), args=[])
  mapping: Dict[str, Any] = {
    "layout_map": {
      "a": "NCHW->NHWC",
    }
  }
  details: Dict[str, Any] = {"std_args": ["a"]}
  with patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_compute:
    mock_compute.return_value = (0, 2, 3, 1)
    result: cst.Call = _apply_layout_permutation(node, mapping, details, rewriter)
    assert len(result.args) == 0
