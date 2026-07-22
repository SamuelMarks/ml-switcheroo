"""Test suite for the Calls Strategy Utils2 module."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import (
  rewrite_stateful_call,
  inject_kwarg,
  strip_kwarg,
  is_super_call,
  is_builtin,
  log_diff,
  compute_permutation,
  inject_permute_call,
  is_functional_apply,
)
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRewriter:
  """Mock Rewriter class for testing purposes."""

  def __init__(self, strict_mode=False):
    """Initializes the MockRewriter instance."""
    self.source_fw = "src"
    self.target_fw = "jax"
    self.strict_mode = strict_mode
    self.failures = []
    self._is_module_alias = lambda x, y: False
    self.context = type(
      "Ctx",
      (),
      {
        "semantics": type("Sem", (), {"resolve_variant": lambda x, y: None, "get_framework_config": lambda x: {}})(),
        "target_framework": "jax",
        "current_file_path": "",
        "config": type("Cfg", (), {"strict_mode": strict_mode})(),
      },
    )()

  def _report_failure(self, msg):
    """Mock implementation of  report failure."""
    self.failures.append(msg)

  def _create_name_node(self, api):
    """Mock implementation of  create name node."""
    import libcst as cst

    return cst.Name(api)


class MockSigCtx:
  """Mock Sig Ctx class for testing purposes."""

  def __init__(self, node=None):
    """Initializes the MockSigCtx instance."""
    self.node = node
    self.existing_args = []
    self.injected_args = []


def test_inject_strip_kwarg():
  """Injects strip keyword argument."""
  orig = cst.Call(func=cst.Name("foo"), args=[])
  res1 = inject_kwarg(orig, "my_kw", "my_val")
  assert len(res1.args) == 1
  assert res1.args[0].keyword.value == "my_kw"
  res2 = inject_kwarg(res1, "my_kw", "other")
  assert res2 is res1
  res3 = strip_kwarg(res1, "my_kw")
  assert len(res3.args) == 0


def test_compute_permutation():
  """Computes permutation."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("AB", "CBA") is None
  assert compute_permutation("AB", "AC") is None


def test_inject_permute_call():
  """Injects permute call."""
  base = cst.Name("x")
  semantics = MagicMock(spec=SemanticsManager)
  semantics.resolve_variant.return_value = None
  res_miss = inject_permute_call(base, (0, 1), semantics, "fw")
  assert res_miss is base
  semantics.resolve_variant.return_value = {"api": "np.transpose", "pack_to_tuple": "axes"}
  res_tuple = inject_permute_call(base, (1, 0), semantics, "fw")
  assert isinstance(res_tuple, cst.Call)
  assert res_tuple.args[1].keyword.value == "axes"
  assert isinstance(res_tuple.args[1].value, cst.Tuple)
  semantics.resolve_variant.return_value = {"api": "torch.permute"}
  res_pos = inject_permute_call(base, (1, 0), semantics, "fw")
  assert isinstance(res_pos, cst.Call)
  assert len(res_pos.args) == 3


def test_is_functional_apply():
  """Checks if is functional apply."""
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), None) is False
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), "apply") is False
  assert is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("apply"))), "apply") is True
  assert (
    is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("other"))), "apply") is False
  )


def test_is_super_call():
  """Checks if is super call."""
  assert is_super_call(cst.Call(func=cst.Name("super"), args=[])) is True
  assert (
    is_super_call(cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("method")))) is True
  )
  assert is_super_call(cst.Call(func=cst.Name("foo"))) is False


def test_is_builtin():
  """Checks if is builtin."""
  assert is_builtin("print") is True
  assert is_builtin("foo") is False


@patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes")
@patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer")
def test_log_diff(mock_get_tracer, mock_diff_nodes):
  """Verifies the behavior of log diff."""
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
  """Injects keyword argument comma."""
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res = inject_kwarg(orig, "kw", "val")
  assert len(res.args) == 2


def test_strip_kwarg_comma():
  """Verifies the behavior of strip keyword argument comma."""
  orig = cst.Call(
    func=cst.Name("foo"),
    args=[cst.Arg(value=cst.Name("x"), comma=cst.Comma()), cst.Arg(keyword=cst.Name("kw"), value=cst.Name("y"))],
  )
  res = strip_kwarg(orig, "kw")
  assert len(res.args) == 1
  assert res.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_execute_strategy_infix_inner_fail():
  """Executes strategy infix inner fail."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping = {"transformation_type": "infix", "operator": "???"}
  details = {"std_args": ["x"]}
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd


def test_execute_strategy_lambda_inner_fail():
  """Executes strategy lambda inner fail."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping = {"transformation_type": "inline_lambda", "api": "lambda: ????"}
  details = {}
  res_fail = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")
  assert res_fail is upd


@patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules")
def test_execute_strategy_dispatch_rules(mock_eval):
  """Executes strategy dispatch rules."""
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
  """Executes strategy strict mode."""
  mock_apply.return_value = [cst.Arg(value=cst.Name("x"))]
  rewriter = MockRewriter(strict_mode=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = orig
  mapping = {"api": "target_foo"}
  res = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")
  assert isinstance(res, cst.Call)
  mock_apply.assert_called_once()


def test_rewrite_stateful_call_no_create_dotted_fixed():
  """Rewrites stateful call no create dotted fixed."""

  class DummyContext:
    """Dummy Context class for testing purposes."""

    def __init__(self):
      """Initializes the DummyContext instance."""
      self.signature_stack = [MockSigCtx()]

  class BasicRewriter:
    """Test suite for the Basic Rewriter component."""

    def __init__(self):
      """Initializes the BasicRewriter instance."""
      self.context = DummyContext()

  rewriter = BasicRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  res = rewrite_stateful_call(rewriter, orig, "my_inst", {"method": "apply"})
  assert res.func.value.value == "my_inst"
