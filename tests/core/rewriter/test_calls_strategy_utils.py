"""Test suite for the Calls Strategy Utils module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as _cst

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import rewrite_stateful_call

cst: typing.Any = _cst


class MockSigCtx:
  """Test helper class MockSigCtx."""

  def __init__(
    self,
    existing: typing.Any = None,
    injected: typing.Any = None,
  ) -> None:
    """Initializes the MockSigCtx instance.

    Args:
        existing: Test argument or fixture.
        injected: Test argument or fixture.
    """
    self.existing_args = existing or []
    self.injected_args = injected or []


class MockContext:
  """Test helper class MockContext."""

  def __init__(self, has_hook: typing.Any = True, has_sig: typing.Any = True) -> None:
    """Initializes the MockContext instance.

    Args:
        has_hook: Test argument or fixture.
        has_sig: Test argument or fixture.
    """
    if has_hook:
      self.hook_context = MagicMock()
    if has_sig:
      self.signature_stack = [MockSigCtx()]


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(
    self,
    fail_norm: typing.Any = False,
    no_imports: typing.Any = False,
    legacy_sig: typing.Any = False,
    strict_mode: typing.Any = False,
  ) -> None:
    """Initializes the MockRewriter instance.

    Args:
        fail_norm: Test argument or fixture.
        no_imports: Test argument or fixture.
        legacy_sig: Test argument or fixture.
        strict_mode: Test argument or fixture.
    """
    self.context = MockContext(has_sig=not legacy_sig)
    if legacy_sig:
      self._signature_stack = [MockSigCtx()]
    self.target_fw = "target"
    self.source_fw = "src"
    self.semantics = MagicMock()
    self.strict_mode = strict_mode
    self.failures: typing.Any = []
    self.warnings: typing.Any = []
    self.fail_norm = fail_norm
    self._is_module_alias: typing.Any = lambda x, y: False

  def _normalize_arguments(self, orig: typing.Any, upd: typing.Any, det: typing.Any, map: typing.Any) -> typing.Any:
    """Mock implementation of  normalize arguments.

    Args:
        orig: Test argument or fixture.
        upd: Test argument or fixture.
        det: Test argument or fixture.
        map: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if self.fail_norm:
      raise ValueError("norm fail")
    return list(upd.args)

  def _report_failure(self, msg: typing.Any) -> None:
    """Mock implementation of  report failure.

    Args:
        msg: Test argument or fixture.
    """
    self.failures.append(msg)

  def _report_warning(self, msg: typing.Any) -> None:
    """Mock implementation of  report warning.

    Args:
        msg: Test argument or fixture.
    """
    self.warnings.append(msg)

  def _create_name_node(self, name: typing.Any) -> typing.Any:
    """Mock implementation of  create name node.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(name)

  def _create_dotted_name(self, name: typing.Any) -> typing.Any:
    """Mock implementation of  create dotted name.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(name)


def test_execute_strategy_infix() -> None:
  """Executes strategy infix."""
  rewriter: typing.Any = MockRewriter()
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: typing.Any = {"transformation_type": "infix", "operator": "-"}
  details: typing.Any = {"std_args": ["x"]}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.UnaryOperation)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
    assert res_fail is upd
    assert len(rewriter.failures) == 1


def test_execute_strategy_inline_lambda() -> None:
  """Executes strategy inline lambda."""
  rewriter: typing.Any = MockRewriter()
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: typing.Any = {"transformation_type": "inline_lambda", "api": "lambda a: a + 1"}
  details: typing.Any = {}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.Call)
  mapping["api"] = "lambda a: +++"
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook: typing.Any) -> None:
  """Executes strategy plugin.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  rewriter: typing.Any = MockRewriter()
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[])
  upd: typing.Any = cst.Call(func=cst.Name("foo"), args=[])
  mock_hook: typing.Any = MagicMock()
  mock_hook.return_value = cst.Name("plugin_res")
  mock_get_hook.return_value = mock_hook
  mapping: typing.Any = {"requires_plugin": "my_plugin"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Name)
  mock_get_hook.return_value = None
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_macro() -> None:
  """Executes strategy macro."""
  rewriter: typing.Any = MockRewriter()
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: typing.Any = {"macro_template": "{x} * 2"}
  details: typing.Any = {"std_args": ["x", ["y", "Y"], {"name": "z"}]}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.BinaryOperation)
  mapping["macro_template"] = "{x} * +++"
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_standard() -> None:
  """Executes strategy standard."""
  rewriter: typing.Any = MockRewriter()
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd: typing.Any = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res_miss: typing.Any = execute_strategy(rewriter, orig, upd, {}, {}, "op_id")
  assert res_miss is upd
  assert len(rewriter.failures) == 1
  mapping: typing.Any = {"api": "target_foo"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  assert typing.cast(cst.Name, res.func).value == "target_foo"
  mapping["layout_map"] = {"x": "NCHW -> NHWC", "return": "NHWC -> NCHW"}
  details: typing.Any = {"std_args": ["x", {"name": "y"}, ["z"]]}
  with (
    patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_comp,
    patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inj,
  ):
    mock_comp.return_value = (0, 2, 3, 1)
    mock_inj.return_value = cst.Name("permuted")
    res_layout: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
    assert mock_comp.call_count == 2
    assert mock_inj.call_count == 2
    assert isinstance(res_layout, cst.Name)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_norm_fail: typing.Any = execute_strategy(rewriter, orig, upd, {"api": "foo"}, {}, "op_id")
    assert res_norm_fail is upd
  assert len(rewriter.failures) > 1


def test_rewrite_stateful_call() -> None:
  """Rewrites stateful call."""
  orig: typing.Any = cst.Call(func=cst.Name("foo"), args=[])
  rewriter: typing.Any = MockRewriter()
  res: typing.Any = rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars", "method": "apply"})
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert typing.cast(cst.Name, res.args[0].value).value == "vars"
  assert res.func.attr.value == "apply"
  assert len(rewriter.warnings) == 1
  rewriter.context.signature_stack[0].injected_args.append(("vars", None))
  rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter.warnings) == 1
  rewriter_leg: typing.Any = MockRewriter(legacy_sig=True)
  rewrite_stateful_call(rewriter_leg, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter_leg.warnings) == 1
  res3: typing.Any = rewrite_stateful_call(rewriter, orig, "my_inst", {})
  assert res3.func == orig.func
