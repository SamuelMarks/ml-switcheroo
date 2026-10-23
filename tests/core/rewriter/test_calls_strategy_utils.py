"""Test suite for the Calls Strategy Utils module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as _cst

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import rewrite_stateful_call

cst = _cst


class MockSigCtx:
  """Test helper class MockSigCtx."""

  def __init__(
    self,
    existing: list[str] | None = None,
    injected: list[tuple[str, str | None]] | None = None,
  ) -> None:
    """Initializes the MockSigCtx instance.

    Args:
        existing: The existing (list[str] | None) for init.
        injected: The injected (list[tuple[str, str | None]] | None) for init.
    """
    self.existing_args = existing or []
    self.injected_args = injected or []


class MockContext:
  """Test helper class MockContext."""

  def __init__(self, has_hook: bool = True, has_sig: bool = True) -> None:
    """Initializes the MockContext instance.

    Args:
        has_hook: The has hook (bool) for init.
        has_sig: The has sig (bool) for init.
    """
    if has_hook:
      self.hook_context = MagicMock()
    if has_sig:
      self.signature_stack = [MockSigCtx()]


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(
    self,
    fail_norm: bool = False,
    no_imports: bool = False,
    legacy_sig: bool = False,
    strict_mode: bool = False,
  ) -> None:
    """Initializes the MockRewriter instance.

    Args:
        fail_norm: The fail norm (bool) for init.
        no_imports: The no imports (bool) for init.
        legacy_sig: The legacy sig (bool) for init.
        strict_mode: The strict mode (bool) for init.
    """
    self.context = MockContext(has_sig=not legacy_sig)
    if legacy_sig:
      self._signature_stack = [MockSigCtx()]
    self.target_framework = "target"
    self.source_framework = "src"
    self.semantics = MagicMock()
    self.strict_mode = strict_mode
    self.failures: list[str] = []
    self.warnings: list[str] = []
    self.fail_norm = fail_norm
    self._is_module_alias = lambda x, y=None: False

  def _normalize_arguments(
    self,
    orig: cst.Call,
    upd: cst.Call,
    det: dict[str, JsonValue],
    map: dict[str, JsonValue],
  ) -> list[cst.Arg]:
    """Mock implementation of  normalize arguments.

    Args:
        orig: The orig (cst.Call) for normalize arguments.
        upd: The upd (cst.Call) for normalize arguments.
        det: The det (dict[str, JsonValue]) for normalize arguments.
        map: The map (dict[str, JsonValue]) for normalize arguments.

    Returns:
        Resulting `list[cst.Arg]` from normalize arguments.
    """
    if self.fail_norm:
      raise ValueError("norm fail")
    return list(upd.args)

  def _report_failure(self, msg: str) -> None:
    """Mock implementation of  report failure.

    Args:
        msg: The msg (str) for report failure.
    """
    self.failures.append(msg)

  def _report_warning(self, msg: str) -> None:
    """Mock implementation of  report warning.

    Args:
        msg: The msg (str) for report warning.
    """
    self.warnings.append(msg)

  def _create_name_node(self, name: str) -> cst.BaseExpression:
    """Mock implementation of  create name node.

    Returns:
        Resulting `cst.BaseExpression` from create name node.
    """
    return cst.Name(name)

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Mock implementation of  create dotted name.

    Returns:
        Resulting `cst.BaseExpression` from create dotted name.
    """
    return cst.Name(name)


def test_execute_strategy_infix() -> None:
  """Executes strategy infix."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {"transformation_type": "infix", "operator": "-"}
  details = {"std_args": ["x"]}
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.UnaryOperation)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
    assert res_fail is upd
    assert len(rewriter.failures) == 1


def test_execute_strategy_inline_lambda() -> None:
  """Executes strategy inline lambda."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {"transformation_type": "inline_lambda", "api": "lambda a: a + 1"}
  details: dict[str, RuntimeTensorOrValue] = {}
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.Call)
  mapping["api"] = "lambda a: +++"
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook: MagicMock) -> None:
  """Executes strategy plugin.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test execute strategy plugin.
  """
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = cst.Call(func=cst.Name("foo"), args=[])
  mock_hook = MagicMock()
  mock_hook.return_value = cst.Name("plugin_res")
  mock_get_hook.return_value = mock_hook
  mapping = {"requires_plugin": "my_plugin"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Name)
  mock_get_hook.return_value = None
  res_fail = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_macro() -> None:
  """Executes strategy macro."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  mapping = {"macro_template": "{x} * 2"}
  details = {"std_args": ["x", ["y", "Y"], {"name": "z"}]}
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.BinaryOperation)
  mapping["macro_template"] = "{x} * +++"
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_standard() -> None:
  """Executes strategy standard."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res_miss = execute_strategy(rewriter, orig, upd, {}, {}, "op_id")
  assert res_miss is upd
  assert len(rewriter.failures) == 1
  mapping = {"api": "target_foo"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  assert cst.ensure_type(res.func, cst.Name).value == "target_foo"
  getattr(getattr({"v": mapping}, "__getitem__")("v"), "__setitem__")(
    "layout_map",
    getattr(
      {
        "v": {
          "x": getattr({"v": "NCHW -> NHWC"}, "__getitem__")("v"),
          "return": getattr({"v": "NHWC -> NCHW"}, "__getitem__")("v"),
        }
      },
      "__getitem__",
    )("v"),
  )
  details = {"std_args": ["x", {"name": "y"}, ["z"]]}
  with (
    patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_comp,
    patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inj,
  ):
    mock_comp.return_value = (0, 2, 3, 1)
    mock_inj.return_value = cst.Name("permuted")
    res_layout = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
    assert mock_comp.call_count == 2
    assert mock_inj.call_count == 2
    assert isinstance(res_layout, cst.Name)
  with patch("ml_switcheroo.core.rewriter.calls.strategy.normalize_arguments", side_effect=ValueError("Norm fail")):
    res_norm_fail = execute_strategy(rewriter, orig, upd, {"api": "foo"}, {}, "op_id")
    assert res_norm_fail is upd
  assert len(rewriter.failures) > 1


def test_rewrite_stateful_call() -> None:
  """Rewrites stateful call."""
  orig = cst.Call(func=cst.Name("foo"), args=[])
  rewriter = MockRewriter()
  res = rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars", "method": "apply"})
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert cst.ensure_type(res.args[0].value, cst.Name).value == "vars"
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": res}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "apply"}, "__getitem__")("v")
  assert len(rewriter.warnings) == 1
  rewriter.context.signature_stack[0].injected_args.append(("vars", None))
  rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter.warnings) == 1
  rewriter_leg = MockRewriter(legacy_sig=True)
  rewrite_stateful_call(rewriter_leg, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter_leg.warnings) == 1
  res3 = rewrite_stateful_call(rewriter, orig, "my_inst", {})
  assert res3.func == orig.func
