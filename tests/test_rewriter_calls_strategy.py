"""Test module."""

import pytest
import libcst as cst
from unittest.mock import Mock, patch

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy, _apply_layout_permutation


class MockRewriter:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.context = Mock()
    self.source_fw = "source_fw"
    self.target_fw = "target_fw"
    self.strict_mode = False
    self._is_module_alias = Mock(return_value=False)
    self.failures = []
    self.semantics = Mock()
    self.semantics._key_origins = {"some_id": "neural"}

  def _report_failure(self, msg):
    self.failures.append(msg)

  def _create_name_node(self, name):
    return cst.Name(name)


@pytest.fixture
def original_call():
  """Test element."""
  return cst.Call(func=cst.Name("foo"), args=[])


@pytest.fixture
def updated_call():
  """Test element."""
  return cst.Call(func=cst.Name("bar"), args=[])


def test_execute_strategy_dispatch_rules(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"dispatch_rules": ["rule1"]}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = "dispatched_api"
    # We need to test how it handles when the standard strategy is executed after dispatch
    mapping["api"] = "original_api"
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")

    mock_eval.assert_called_once_with(rewriter, original_call, ["rule1"], details)
    assert getattr(result.func, "value", None) == "dispatched_api"


def test_execute_strategy_variant_imports(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  rewriter._handle_variant_imports = Mock()
  mapping = {"api": "foo"}
  execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")
  rewriter._handle_variant_imports.assert_called_once_with(mapping)


def test_execute_strategy_infix_success(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"transformation_type": "infix", "operator": "+"}
  details = {"std_args": ["a", "b"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))]
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.BinaryOperation)


def test_execute_strategy_infix_failure(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"transformation_type": "infix", "operator": "+"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Infix/Prefix transformation failed:" in rewriter.failures[0]


def test_execute_strategy_inline_lambda_success(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = []
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert isinstance(result, cst.Call)


def test_execute_strategy_inline_lambda_failure(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Inline lambda transformation failed:" in rewriter.failures[0]


def test_execute_strategy_plugin_success(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"requires_plugin": "my_plugin"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_hook = Mock(return_value="plugin_result")
    mock_get_hook.return_value = mock_hook
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == "plugin_result"
    mock_hook.assert_called_once_with(updated_call, rewriter.context.hook_context)


def test_execute_strategy_plugin_failure(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"requires_plugin": "my_plugin"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook") as mock_get_hook:
    mock_get_hook.return_value = None
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Missing required plugin:" in rewriter.failures[0]


def test_execute_strategy_macro_success(original_call, updated_call):
  """Test element."""
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


def test_execute_strategy_macro_failure(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"macro_template": "{x} + 1"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = Exception("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Macro expansion failed:" in rewriter.failures[0]


def test_execute_strategy_standard_missing_api(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {}
  details = {}

  result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert "No mapping available" in rewriter.failures[0]


def test_execute_strategy_standard_missing_api_neural(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  rewriter.target_fw = "jax"
  mapping = {}
  details = {}

  result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
  assert result == updated_call
  assert "Cannot map neural network abstraction" in rewriter.failures[0]


def test_execute_strategy_standard_success(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  rewriter.strict_mode = True
  mapping = {"api": "new_api"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards") as mock_guards:
      mock_guards.return_value = [cst.Arg(value=cst.Name("a"))]
      result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert isinstance(result, cst.Call)
      assert result.func.value == "new_api"


def test_execute_strategy_standard_layout_map(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"api": "new_api", "layout_map": {"a": "NCHW->NHWC"}}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.return_value = [cst.Arg(value=cst.Name("a"))]
    with patch("ml_switcheroo.core.rewriter.calls.strategy._apply_layout_permutation") as mock_layout:
      mock_layout.return_value = "layout_result"
      result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
      assert result == "layout_result"


def test_execute_strategy_standard_normalization_failure(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"api": "new_api"}
  details = {}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments") as mock_norm:
    mock_norm.side_effect = ValueError("error")
    result = execute_strategy(rewriter, original_call, updated_call, mapping, details, "some_id")
    assert result == updated_call
    assert "Argument normalization failed" in rewriter.failures[0]


def test_apply_layout_permutation():
  """Test element."""
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
      assert result.value == "permuted_return"


def test_execute_strategy_no_hook_context(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  del rewriter.context.hook_context
  mapping = {"api": "foo"}
  execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")


def test_execute_strategy_dispatch_rules_falsy(original_call, updated_call):
  """Test element."""
  rewriter = MockRewriter()
  mapping = {"dispatch_rules": ["rule1"]}

  with patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules") as mock_eval:
    mock_eval.return_value = None
    execute_strategy(rewriter, original_call, updated_call, mapping, {}, "some_id")


def test_apply_layout_permutation_no_arrow():
  """Test element."""
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


def test_apply_layout_permutation_falsy_indices():
  """Test element."""
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


def test_apply_layout_permutation_out_of_bounds_idx():
  """Test element."""
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
    assert len(result.args) == 0
