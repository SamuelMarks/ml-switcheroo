"""Module docstring."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy, _apply_layout_permutation
from ml_switcheroo.core.rewriter.calls.utils import (
  rewrite_stateful_call,
  inject_kwarg,
  strip_kwarg,
  is_super_call,
  is_builtin,
  log_diff,
  compute_permutation,
  inject_permute_call,
)
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSigCtx:
  """Class docstring."""

  def __init__(self, existing=None, injected=None):
    """Function docstring."""
    self.existing_args = existing or []
    self.injected_args = injected or []


class MockContext:
  """Class docstring."""

  def __init__(self, has_hook=True, has_sig=True):
    """Function docstring."""
    if has_hook:
      self.hook_context = MagicMock()
    if has_sig:
      self.signature_stack = [MockSigCtx()]


class MockRewriter:
  """Class docstring."""

  def __init__(self, fail_norm=False, no_imports=False, legacy_sig=False, strict_mode=False):
    """Function docstring."""
    self.context = MockContext(has_sig=not legacy_sig)
    if legacy_sig:
      self._signature_stack = [MockSigCtx()]
    self.target_fw = "target"
    self.semantics = MagicMock()
    self.strict_mode = strict_mode
    self.failures = []
    self.warnings = []
    self.fail_norm = fail_norm
    self.no_imports = no_imports
    if not no_imports:
      self._handle_variant_imports = MagicMock()

  def _normalize_arguments(self, orig, upd, det, map):
    """Function docstring."""
    if self.fail_norm:
      raise ValueError("norm fail")
    return list(upd.args)

  def _report_failure(self, msg):
    """Function docstring."""
    self.failures.append(msg)

  def _report_warning(self, msg):
    """Function docstring."""
    self.warnings.append(msg)

  def _create_name_node(self, name):
    """Function docstring."""
    return cst.Name(name)

  def _create_dotted_name(self, name):
    """Function docstring."""
    return cst.Name(name)


# --- test strategy.py ---


def test_execute_strategy_infix():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])

  mapping = {"transformation_type": "infix", "operator": "-"}
  details = {"std_args": ["x"]}
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.UnaryOperation)

  # Infix fail
  rewriter.fail_norm = True
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_inline_lambda():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])

  mapping = {"transformation_type": "inline_lambda", "api": "lambda a: a + 1"}
  details = {}
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.Call)

  # Lambda fail
  mapping["api"] = "lambda a: +++"
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


@patch("ml_switcheroo.core.rewriter.calls.strategy.get_hook")
def test_execute_strategy_plugin(mock_get_hook):
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = cst.Call(func=cst.Name("foo"), args=[])

  mock_hook = MagicMock()
  mock_hook.return_value = cst.Name("plugin_res")
  mock_get_hook.return_value = mock_hook

  mapping = {"requires_plugin": "my_plugin"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Name)

  # Plugin missing
  mock_get_hook.return_value = None
  res_fail = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_macro():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])

  mapping = {"macro_template": "{x} * 2"}
  details = {"std_args": ["x", ["y", "Y"], {"name": "z"}]}  # Test parsing of std_args items
  res = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert isinstance(res, cst.BinaryOperation)

  # Macro fail
  mapping["macro_template"] = "{x} * +++"
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd
  assert len(rewriter.failures) == 1


def test_execute_strategy_standard():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])

  # Missing api
  res_miss = execute_strategy(rewriter, orig, upd, {}, {}, "op_id")
  assert res_miss is upd
  assert len(rewriter.failures) == 1

  # Normal success
  mapping = {"api": "target_foo"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  assert res.func.value == "target_foo"

  # Layout permutation
  mapping["layout_map"] = {"x": "NCHW -> NHWC", "return": "NHWC -> NCHW"}
  details = {"std_args": ["x", {"name": "y"}, ["z"]]}

  # Mock compute_permutation and inject_permute_call
  with (
    patch("ml_switcheroo.core.rewriter.calls.strategy.compute_permutation") as mock_comp,
    patch("ml_switcheroo.core.rewriter.calls.strategy.inject_permute_call") as mock_inj,
  ):
    mock_comp.return_value = (0, 2, 3, 1)
    mock_inj.return_value = cst.Name("permuted")

    res_layout = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
    assert mock_comp.call_count == 2
    assert mock_inj.call_count == 2
    assert isinstance(res_layout, cst.Name)  # return permutation wrapped the whole node

  # Argument normalization failure
  rewriter.fail_norm = True
  res_norm_fail = execute_strategy(rewriter, orig, upd, {"api": "foo"}, {}, "op_id")
  assert res_norm_fail is upd
  assert len(rewriter.failures) > 1


# --- test utils.py ---


def test_rewrite_stateful_call():
  """Function docstring."""
  orig = cst.Call(func=cst.Name("foo"), args=[])

  # Normal Context
  rewriter = MockRewriter()
  res = rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars", "method": "apply"})
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert res.args[0].value.value == "vars"
  assert res.func.attr.value == "apply"
  assert len(rewriter.warnings) == 1

  # Test existing injected arg (no warning)
  rewriter.context.signature_stack[0].injected_args.append(("vars", None))
  res2 = rewrite_stateful_call(rewriter, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter.warnings) == 1  # unchanged

  # Legacy Context
  rewriter_leg = MockRewriter(legacy_sig=True)
  rewrite_stateful_call(rewriter_leg, orig, "my_inst", {"prepend_arg": "vars"})
  assert len(rewriter_leg.warnings) == 1

  # No method fallback
  res3 = rewrite_stateful_call(rewriter, orig, "my_inst", {})
  assert res3.func == orig.func


def test_inject_strip_kwarg():
  """Function docstring."""
  orig = cst.Call(func=cst.Name("foo"), args=[])

  # Inject
  res1 = inject_kwarg(orig, "my_kw", "my_val")
  assert len(res1.args) == 1
  assert res1.args[0].keyword.value == "my_kw"

  # Inject existing
  res2 = inject_kwarg(res1, "my_kw", "other")
  assert res2 is res1

  # Strip
  res3 = strip_kwarg(res1, "my_kw")
  assert len(res3.args) == 0


def test_compute_permutation():
  """Function docstring."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("AB", "CBA") is None
  assert compute_permutation("AB", "AC") is None


def test_inject_permute_call():
  """Function docstring."""
  base = cst.Name("x")
  semantics = MagicMock(spec=SemanticsManager)

  # Missing definition
  semantics.resolve_variant.return_value = None
  res_miss = inject_permute_call(base, (0, 1), semantics, "fw")
  assert res_miss is base

  # Pack to tuple
  semantics.resolve_variant.return_value = {"api": "np.transpose", "pack_to_tuple": "axes"}
  res_tuple = inject_permute_call(base, (1, 0), semantics, "fw")
  assert isinstance(res_tuple, cst.Call)
  assert res_tuple.args[1].keyword.value == "axes"
  assert isinstance(res_tuple.args[1].value, cst.Tuple)

  # Positional
  semantics.resolve_variant.return_value = {"api": "torch.permute"}
  res_pos = inject_permute_call(base, (1, 0), semantics, "fw")
  assert isinstance(res_pos, cst.Call)
  assert len(res_pos.args) == 3  # base, 1, 0


from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply


def test_is_functional_apply():
  """Function docstring."""
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), None) is False
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), "apply") is False
  assert is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("apply"))), "apply") is True
  assert (
    is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("other"))), "apply") is False
  )


def test_is_super_call():
  """Function docstring."""
  assert is_super_call(cst.Call(func=cst.Name("super"), args=[])) is True
  assert (
    is_super_call(cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("method")))) is True
  )
  assert is_super_call(cst.Call(func=cst.Name("foo"))) is False


def test_is_builtin():
  """Function docstring."""
  assert is_builtin("print") is True
  assert is_builtin("foo") is False


@patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes")
@patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer")
def test_log_diff(mock_get_tracer, mock_diff_nodes):
  """Function docstring."""
  mock_diff_nodes.return_value = ("a", "b", True)
  mock_tracer = MagicMock()
  mock_get_tracer.return_value = mock_tracer
  log_diff("label", cst.Name("a"), cst.Name("b"))
  mock_tracer.log_mutation.assert_called_once_with("label", "a", "b")

  mock_diff_nodes.return_value = ("a", "a", False)
  mock_tracer.reset_mock()
  log_diff("label", cst.Name("a"), cst.Name("a"))
  mock_tracer.log_mutation.assert_not_called()


def test_inject_kwarg_comma():
  """Function docstring."""
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res = inject_kwarg(orig, "kw", "val")
  assert len(res.args) == 2


def test_strip_kwarg_comma():
  """Function docstring."""
  orig = cst.Call(
    func=cst.Name("foo"),
    args=[cst.Arg(value=cst.Name("x"), comma=cst.Comma()), cst.Arg(keyword=cst.Name("kw"), value=cst.Name("y"))],
  )
  res = strip_kwarg(orig, "kw")
  assert len(res.args) == 1
  assert res.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_execute_strategy_infix_inner_fail():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping = {"transformation_type": "infix", "operator": "???"}  # bad op
  details = {"std_args": ["x"]}
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd


def test_execute_strategy_lambda_inner_fail():
  """Function docstring."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping = {"transformation_type": "inline_lambda", "api": "lambda: ????"}  # syntax error
  details = {}
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd


@patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules")
def test_execute_strategy_dispatch_rules(mock_eval):
  """Function docstring."""
  mock_eval.return_value = "dispatched_target"
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = orig
  mapping = {"dispatch_rules": [{"condition": "test"}], "api": "old_api"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  assert res.func.value == "dispatched_target"


@patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards")
def test_execute_strategy_strict_mode(mock_apply):
  """Function docstring."""
  mock_apply.return_value = [cst.Arg(value=cst.Name("x"))]
  rewriter = MockRewriter(strict_mode=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = orig
  mapping = {"api": "target_foo"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  mock_apply.assert_called_once()


def test_rewrite_stateful_call_no_create_dotted_fixed():
  """Function docstring."""

  # Pass an object that does not have _create_dotted_name
  class DummyContext:
    """Class docstring."""

    def __init__(self):
      """Function docstring."""
      self.signature_stack = [MockSigCtx()]

  class BasicRewriter:
    """Class docstring."""

    def __init__(self):
      """Function docstring."""
      self.context = DummyContext()

  rewriter = BasicRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  res = rewrite_stateful_call(rewriter, orig, "my_inst", {"method": "apply"})
  assert res.func.value.value == "my_inst"
