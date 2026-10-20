"""Test suite for rewriter strategy coverage."""

import typing

import libcst as _cst

from ml_switcheroo.core.hooks_registry import clear_hooks
from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy

cst = _cst


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context = type("HookCtx", (), {"current_op_id": None})()


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = DummyContext()
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.strict_mode = False

    class Semantics:
      """Test helper class Semantics."""

      pass

    self.semantics = Semantics()

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Execute is module alias helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return False

  def _create_name_node(self, name: str) -> cst.BaseExpression:
    """Execute create name node helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(name)

  def _report_failure(self, msg: str) -> None:
    """Execute report failure helper logic.

    Args:
        msg: Test argument or fixture.
    """
    pass


def test_execute_strategy_branches() -> None:
  """Verify execute strategy branches behavior."""
  rewriter = DummyRewriter()
  original = cst.parse_statement("f()").body[0].value  # type: ignore[index, union-attr]
  updated = original
  details: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"variants": {}}

  class DummyRule:
    """Test helper class DummyRule."""

    def __init__(self, if_arg: str, if_value: int, use_api: str) -> None:
      """Initialize test helper instance.

      Args:
          if_arg: Test argument or fixture.
          if_value: Test argument or fixture.
          use_api: Test argument or fixture.
      """
      self.if_arg = if_arg
      self.is_val = if_value
      self.use_api = use_api
      self.op = None

  mapping1 = {"dispatch_rules": [DummyRule(if_arg="x", if_value=1, use_api="f1")], "api": "default"}
  execute_strategy(rewriter, original, updated, mapping1, details, "op1")

  mapping2 = {"transformation_type": "infix", "operator": "+"}
  execute_strategy(rewriter, original, updated, mapping2, details, "op2")

  mapping3 = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  execute_strategy(rewriter, original, updated, mapping3, details, "op3")

  mapping4 = {"requires_plugin": "my_plugin"}
  clear_hooks()
  execute_strategy(rewriter, original, updated, mapping4, details, "op4")

  mapping5 = {"macro_template": "a + b"}
  details_macro = {"std_args": ["a", {"name": "b"}, ("c", "type")]}
  _res = execute_strategy(rewriter, original, updated, mapping5, details_macro, "op5")
  assert isinstance(_res, cst.BinaryOperation)


def test_apply_layout_permutation_branches() -> None:
  """Verify apply layout permutation branches behavior."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("f(a, b)").body[0].value  # type: ignore[index, union-attr]
  details = {"std_args": ["a", "b"]}

  mapping = {"layout_map": {"a": "A -> B"}}
  _res = _apply_layout_permutation(node, mapping, details, rewriter)
  assert isinstance(_res, cst.Call)
  assert len(_res.args) == 2


def test_apply_layout_transformations_no_arrow() -> None:
  """Verify apply layout transformations no arrow behavior."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation

  class DummyRewriter:
    """Test helper class DummyRewriter."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.semantics = MagicMock()
      self.target_fw = "jax"

  rewriter = DummyRewriter()
  node = cst.parse_statement("f(x)").body[0].value  # type: ignore[index, union-attr]

  mapping = {"layout_map": {"x": "NCHW", "return": "NHWC"}}
  details = {"std_args": [{"name": "x"}]}

  # "->" not in rule, so it should bypass both if statements
  res = _apply_layout_permutation(node, mapping, details, rewriter)  # type: ignore[arg-type]
  assert res.args[0].value.value == "x"  # type: ignore[attr-defined]


def test_strategy_missing_api_custom_message() -> None:
  """Verify strategy missing api custom message behavior."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value  # type: ignore[index, union-attr]
  mapping = {"missing_message": "Custom error message"}
  rewriter.last_failure = None  # type: ignore[attr-defined]

  def mock_report_failure(msg: str) -> None:
    """Execute mock report failure helper logic.

    Args:
        msg: Test argument or fixture.
    """
    rewriter.last_failure = msg  # type: ignore[attr-defined]

  setattr(rewriter, "_report_failure", mock_report_failure)
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "Custom error message"  # type: ignore[attr-defined]


def test_strategy_missing_api_default_message() -> None:
  """Verify strategy missing api default message behavior."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value  # type: ignore[index, union-attr]
  mapping: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
  rewriter.last_failure = None  # type: ignore[attr-defined]

  def mock_report_failure(msg: str) -> None:
    """Execute mock report failure helper logic.

    Args:
        msg: Test argument or fixture.
    """
    rewriter.last_failure = msg  # type: ignore[attr-defined]

  setattr(rewriter, "_report_failure", mock_report_failure)
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "No mapping available for 'foo' -> 'jax'"  # type: ignore[attr-defined]


def test_strategy_neural_rejection() -> None:
  """Verify strategy neural rejection behavior."""
  rewriter = DummyRewriter()
  rewriter.target_fw = "jax"
  rewriter.semantics._key_origins = {"op1": "neural"}  # type: ignore[attr-defined]
  call = cst.parse_statement("f()").body[0].value  # type: ignore[index, union-attr]
  mapping: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}  # No api key
  rewriter.last_failure = None  # type: ignore[attr-defined]

  def mock_report_failure(msg: str) -> None:
    """Execute mock report failure helper logic.

    Args:
        msg: Test argument or fixture.
    """
    rewriter.last_failure = msg  # type: ignore[attr-defined]

  setattr(rewriter, "_report_failure", mock_report_failure)
  execute_strategy(rewriter, call, call, mapping, {}, "op1")
  assert "Cannot map neural network abstraction" in rewriter.last_failure  # type: ignore[attr-defined]
