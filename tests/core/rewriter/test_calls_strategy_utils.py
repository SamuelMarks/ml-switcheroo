"""Test suite for the Calls Strategy Utils module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import rewrite_stateful_call


class MockSigCtx:
  """Docstring."""

  def __init__(
    self,
    existing: typing.Optional[list[str]] = None,
    injected: typing.Optional[list[tuple[str, typing.Optional[cst.CSTNode]]]] = None,
  ) -> None:
    """Initializes the MockSigCtx instance."""
    self.existing_args = existing or []
    self.injected_args = injected or []


class MockContext:
  """Docstring."""

  def __init__(self, has_hook: bool = True, has_sig: bool = True) -> None:
    """Initializes the MockContext instance."""
    if has_hook:
      self.hook_context = MagicMock()
    if has_sig:
      self.signature_stack = [MockSigCtx()]


class MockRewriter:
  """Docstring."""

  def __init__(
    self, fail_norm: bool = False, no_imports: bool = False, legacy_sig: bool = False, strict_mode: bool = False
  ) -> None:
    """Initializes the MockRewriter instance."""
    self.context = MockContext(has_sig=not legacy_sig)
    if legacy_sig:
      self._signature_stack = [MockSigCtx()]
    self.target_fw = "target"
    self.source_fw = "src"
    self.semantics = MagicMock()
    self.strict_mode = strict_mode
    self.failures: list[str] = []
    self.warnings: list[str] = []
    self.fail_norm = fail_norm
    self._is_module_alias: typing.Callable[[typing.Any, typing.Any], bool] = lambda x, y: False

  def _normalize_arguments(self, orig: typing.Any, upd: typing.Any, det: typing.Any, map: typing.Any) -> list[cst.Arg]:
    """Mock implementation of  normalize arguments."""
    if self.fail_norm:
      raise ValueError("norm fail")
    return list(upd.args)

  def _report_failure(self, msg: str) -> None:
    """Mock implementation of  report failure."""
    self.failures.append(msg)

  def _report_warning(self, msg: str) -> None:
    """Mock implementation of  report warning."""
    self.warnings.append(msg)

  def _create_name_node(self, name: str) -> cst.Name:
    """Mock implementation of  create name node."""
    return cst.Name(name)

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Mock implementation of  create dotted name."""
    return cst.Name(name)


def test_execute_strategy_infix() -> None:
  """Executes strategy infix."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: dict[str, typing.Any] = {"transformation_type": "infix", "operator": "-"}
  details: dict[str, typing.Any] = {"std_args": ["x"]}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert isinstance(res, cst.UnaryOperation)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
    assert res_fail is upd
    assert len(rewriter.failures) == 1


def test_execute_strategy_inline_lambda() -> None:
  """Executes strategy inline lambda."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: dict[str, typing.Any] = {"transformation_type": "inline_lambda", "api": "lambda a: a + 1"}
  details: dict[str, typing.Any] = {}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert isinstance(res, cst.Call)
  mapping["api"] = "lambda a: +++"
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert res_fail is upd
  assert len(rewriter.failures) == 1


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook: MagicMock) -> None:
  """Executes strategy plugin."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = cst.Call(func=cst.Name("foo"), args=[])
  mock_hook = MagicMock()
  mock_hook.return_value = cst.Name("plugin_res")
  mock_get_hook.return_value = mock_hook
  mapping: dict[str, typing.Any] = {"requires_plugin": "my_plugin"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")  # type: ignore
  assert isinstance(res, cst.Name)
  mock_get_hook.return_value = None
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")  # type: ignore
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_macro() -> None:
  """Executes strategy macro."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping: dict[str, typing.Any] = {"macro_template": "{x} * 2"}
  details: dict[str, typing.Any] = {"std_args": ["x", ["y", "Y"], {"name": "z"}]}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert isinstance(res, cst.BinaryOperation)
  mapping["macro_template"] = "{x} * +++"
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_standard() -> None:
  """Executes strategy standard."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res_miss: typing.Any = execute_strategy(rewriter, orig, upd, {}, {}, "op_id")  # type: ignore
  assert res_miss is upd
  assert len(rewriter.failures) == 1
  mapping: dict[str, typing.Any] = {"api": "target_foo"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")  # type: ignore
  assert isinstance(res, cst.Call)
  assert typing.cast(cst.Name, res.func).value == "target_foo"
  mapping["layout_map"] = {"x": "NCHW -> NHWC", "return": "NHWC -> NCHW"}
  details: dict[str, typing.Any] = {"std_args": ["x", {"name": "y"}, ["z"]]}
  with (
    patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_comp,
    patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inj,
  ):
    mock_comp.return_value = (0, 2, 3, 1)
    mock_inj.return_value = cst.Name("permuted")
    res_layout: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
    assert mock_comp.call_count == 2
    assert mock_inj.call_count == 2
    assert isinstance(res_layout, cst.Name)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_norm_fail: typing.Any = execute_strategy(rewriter, orig, upd, {"api": "foo"}, {}, "op_id")  # type: ignore
    assert res_norm_fail is upd
  assert len(rewriter.failures) > 1


def test_rewrite_stateful_call() -> None:
  """Rewrites stateful call."""
  orig = cst.Call(func=cst.Name("foo"), args=[])
  rewriter = MockRewriter()
  res: typing.Any = rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars", "method": "apply"})  # type: ignore
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert typing.cast(cst.Name, res.args[0].value).value == "vars"
  assert typing.cast(cst.Name, typing.cast(cst.Attribute, res.func).attr).value == "apply"
  assert len(rewriter.warnings) == 1
  rewriter.context.signature_stack[0].injected_args.append(("vars", None))
  rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars"})  # type: ignore
  assert len(rewriter.warnings) == 1
  rewriter_leg = MockRewriter(legacy_sig=True)
  rewrite_stateful_call(rewriter_leg, orig, "my_inst", {"prepend_arg": "vars"})  # type: ignore
  assert len(rewriter_leg.warnings) == 1
  res3: typing.Any = rewrite_stateful_call(rewriter, orig, "my_inst", {})  # type: ignore
  assert res3.func == orig.func
