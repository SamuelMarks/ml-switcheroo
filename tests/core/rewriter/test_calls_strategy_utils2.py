"""Test suite for the Calls Strategy Utils2 module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_kwarg,
  inject_permute_call,
  is_builtin,
  is_functional_apply,
  is_super_call,
  log_diff,
  rewrite_stateful_call,
  strip_kwarg,
)
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRewriter:
  """Docstring."""

  def __init__(self, strict_mode: bool = False) -> None:
    """Initializes the MockRewriter instance."""
    self.source_fw = "src"
    self.target_fw = "jax"
    self.strict_mode = strict_mode
    self.failures: list[str] = []
    self._is_module_alias: typing.Callable[[typing.Any, typing.Any], bool] = lambda x, y: False
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

  def _report_failure(self, msg: str) -> None:
    """Mock implementation of  report failure."""
    self.failures.append(msg)

  def _create_name_node(self, api: str) -> cst.Name:
    """Mock implementation of  create name node."""
    import libcst as cst

    return cst.Name(api)


class MockSigCtx:
  """Docstring."""

  def __init__(self, node: typing.Any = None) -> None:
    """Initializes the MockSigCtx instance."""
    self.node = node
    self.existing_args: list[str] = []
    self.injected_args: list[tuple[str, typing.Optional[cst.CSTNode]]] = []


def test_inject_strip_kwarg() -> None:
  """Injects strip keyword argument."""
  orig = cst.Call(func=cst.Name("foo"), args=[])
  res1 = inject_kwarg(orig, "my_kw", "my_val")
  assert len(res1.args) == 1
  assert typing.cast(cst.Name, res1.args[0].keyword).value == "my_kw"
  res2 = inject_kwarg(res1, "my_kw", "other")
  assert res2 is res1
  res3 = strip_kwarg(res1, "my_kw")
  assert len(res3.args) == 0


def test_compute_permutation() -> None:
  """Computes permutation."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("AB", "CBA") is None
  assert compute_permutation("AB", "AC") is None


def test_inject_permute_call() -> None:
  """Injects permute call."""
  base = cst.Name("x")
  semantics = MagicMock(spec=SemanticsManager)
  semantics.resolve_variant.return_value = None
  res_miss: typing.Any = inject_permute_call(base, (0, 1), semantics, "fw")  # type: ignore
  assert res_miss is base
  semantics.resolve_variant.return_value = {"api": "np.transpose", "pack_to_tuple": "axes"}
  res_tuple: typing.Any = inject_permute_call(base, (1, 0), semantics, "fw")  # type: ignore
  assert isinstance(res_tuple, cst.Call)
  assert typing.cast(cst.Name, res_tuple.args[1].keyword).value == "axes"
  assert isinstance(res_tuple.args[1].value, cst.Tuple)
  semantics.resolve_variant.return_value = {"api": "torch.permute"}
  res_pos: typing.Any = inject_permute_call(base, (1, 0), semantics, "fw")  # type: ignore
  assert isinstance(res_pos, cst.Call)
  assert len(res_pos.args) == 3


def test_is_functional_apply() -> None:
  """Checks if is functional apply."""
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), None) is False
  assert is_functional_apply(cst.Call(func=cst.Name("foo")), "apply") is False
  assert is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("apply"))), "apply") is True
  assert (
    is_functional_apply(cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("other"))), "apply") is False
  )


def test_is_super_call() -> None:
  """Checks if is super call."""
  assert is_super_call(cst.Call(func=cst.Name("super"), args=[])) is True
  assert (
    is_super_call(cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("method")))) is True
  )
  assert is_super_call(cst.Call(func=cst.Name("foo"))) is False


def test_is_builtin() -> None:
  """Checks if is builtin."""
  assert is_builtin("print") is True
  assert is_builtin("foo") is False


@patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes")
@patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer")
def test_log_diff(mock_get_tracer: MagicMock, mock_diff_nodes: MagicMock) -> None:
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


def test_inject_kwarg_comma() -> None:
  """Injects keyword argument comma."""
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  res = inject_kwarg(orig, "kw", "val")
  assert len(res.args) == 2


def test_strip_kwarg_comma() -> None:
  """Verifies the behavior of strip keyword argument comma."""
  orig = cst.Call(
    func=cst.Name("foo"),
    args=[cst.Arg(value=cst.Name("x"), comma=cst.Comma()), cst.Arg(keyword=cst.Name("kw"), value=cst.Name("y"))],
  )
  res = strip_kwarg(orig, "kw")
  assert len(res.args) == 1
  assert res.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_execute_strategy_infix_inner_fail() -> None:
  """Executes strategy infix inner fail."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping: dict[str, typing.Any] = {"transformation_type": "infix", "operator": "???"}
  details: dict[str, typing.Any] = {"std_args": ["x"]}
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert res_fail is upd


def test_execute_strategy_lambda_inner_fail() -> None:
  """Executes strategy lambda inner fail."""
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("x"))])
  upd = orig
  mapping: dict[str, typing.Any] = {"transformation_type": "inline_lambda", "api": "lambda: ????"}
  details: dict[str, typing.Any] = {}
  res_fail: typing.Any = execute_strategy(rewriter, orig, upd, mapping, details, "op_id")  # type: ignore
  assert res_fail is upd


@patch("ml_switcheroo.core.rewriter.calls.strategy.evaluate_dispatch_rules")
def test_execute_strategy_dispatch_rules(mock_eval: MagicMock) -> None:
  """Executes strategy dispatch rules."""
  mock_eval.return_value = "dispatched_target"
  rewriter = MockRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = orig
  mapping: dict[str, typing.Any] = {"dispatch_rules": [{"condition": "test"}], "api": "old_api"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")  # type: ignore
  assert isinstance(res, cst.Call)
  assert typing.cast(cst.Name, res.func).value == "dispatched_target"


@patch("ml_switcheroo.core.rewriter.calls.strategy.apply_strict_guards")
def test_execute_strategy_strict_mode(mock_apply: MagicMock) -> None:
  """Executes strategy strict mode."""
  mock_apply.return_value = [cst.Arg(value=cst.Name("x"))]
  rewriter = MockRewriter(strict_mode=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  upd = orig
  mapping: dict[str, typing.Any] = {"api": "target_foo"}
  res: typing.Any = execute_strategy(rewriter, orig, upd, mapping, {}, "op_id")  # type: ignore
  assert isinstance(res, cst.Call)
  mock_apply.assert_called_once()


def test_rewrite_stateful_call_no_create_dotted_fixed() -> None:
  """Rewrites stateful call no create dotted fixed."""

  class DummyContext:
    def __init__(self) -> None:
      """Initializes the DummyContext instance."""
      self.signature_stack = [MockSigCtx()]

  class BasicRewriter:
    def __init__(self) -> None:
      """Initializes the BasicRewriter instance."""
      self.context = DummyContext()

  rewriter = BasicRewriter()
  orig = cst.Call(func=cst.Name("foo"), args=[])
  res: typing.Any = rewrite_stateful_call(rewriter, orig, "my_inst", {"method": "apply"})  # type: ignore
  assert typing.cast(cst.Name, typing.cast(cst.Attribute, res.func).value).value == "my_inst"
