"""Test suite for the Schedulers module."""

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.schedulers import (
  _create_dotted_name,
  _get_target_arg_name,
  _transform_cosine_lr,
  _transform_step_lr,
  transform_scheduler_init,
  transform_scheduler_step,
)


class DummyContext(HookContext):
  """Test double `DummyContext` implementing `lookup_api` for schedulers tests."""

  def __init__(
    self,
    op_id: str | None,
    api: str | None,
    variant_args: dict[str, str | float | int | None] | None = None,
  ) -> None:
    """Initializes the DummyContext instance.

    Args:
        op_id: Input `op_id` parameter value (`str | None`).
        api: Input `api` parameter value (`str | None`).
        variant_args: Mapping of standard argument names to target argument names.
    """
    super().__init__(semantics=None, config=RuntimeConfig())
    self.current_op_id = op_id
    self._api: str | None = api
    self._variant: FrameworkVariant | None = FrameworkVariant(args=variant_args if variant_args is not None else {})

  @property
  def current_variant(self) -> FrameworkVariant | None:
    """Return the active framework variant configured for this test context.

    Returns:
        Configured `FrameworkVariant` instance or `None`.
    """
    return self._variant

  @current_variant.setter
  def current_variant(self, value: FrameworkVariant | None) -> None:
    """Set the active framework variant for test scenarios.

    Args:
        value: Replacement `FrameworkVariant` instance or `None`.
    """
    self._variant = value

  def lookup_api(self, op_name: str) -> str | None:
    """Return the configured target API string for the requested operation.

    Args:
        op_name: Standard operation identifier (`str`).

    Returns:
        Configured API string or `None`.
    """
    return self._api


def test_create_dotted_name() -> None:
  """Creates dotted name."""
  node = _create_dotted_name("a")
  assert isinstance(node, cst.Name)
  assert node.value == "a"
  node_attr = _create_dotted_name("a.b.c")
  assert isinstance(node_attr, cst.Attribute)
  assert node_attr.attr.value == "c"
  assert isinstance(node_attr.value, cst.Attribute)
  assert node_attr.value.attr.value == "b"
  assert isinstance(node_attr.value.value, cst.Name)
  assert node_attr.value.value.value == "a"


def test_get_target_arg_name() -> None:
  """Gets target argument name."""
  ctx1: DummyContext = DummyContext("op", "api")
  ctx1.current_variant = None
  assert (
    _get_target_arg_name(
      ctx1,
      "std_name",
      "default",
    )
    == "default"
  )
  ctx2: DummyContext = DummyContext("op", "api", variant_args=None)
  if ctx2.current_variant is not None:
    ctx2.current_variant.args = None
  assert (
    _get_target_arg_name(
      ctx2,
      "std_name",
      "default",
    )
    == "default"
  )
  ctx3: DummyContext = DummyContext("op", "api", variant_args={})
  assert (
    _get_target_arg_name(
      ctx3,
      "std_name",
      "default",
    )
    == "default"
  )
  ctx4: DummyContext = DummyContext("op", "api", variant_args={"std_name": "target_name"})
  assert (
    _get_target_arg_name(
      ctx4,
      "std_name",
      "default",
    )
    == "target_name"
  )


def test_transform_scheduler_init_no_api() -> None:
  """Transforms scheduler initialization no API."""
  ctx: DummyContext = DummyContext("StepLR", None)
  call_node: cst.Call = cst.Call(func=cst.Name("StepLR"))
  result: cst.CSTNode | cst.Call = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_unknown_op() -> None:
  """Transforms scheduler initialization unknown op."""
  ctx: DummyContext = DummyContext("UnknownLR", "target.api")
  call_node: cst.Call = cst.Call(func=cst.Name("UnknownLR"))
  result: cst.CSTNode | cst.Call = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_none_op_id() -> None:
  """Transforms scheduler initialization none op id."""
  ctx: DummyContext = DummyContext(None, "target.api")
  call_node: cst.Call = cst.Call(func=cst.Name("UnknownLR"))
  result: cst.CSTNode | cst.Call = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_step_lr() -> None:
  """Transforms scheduler initialization step lr."""
  ctx: DummyContext = DummyContext("StepLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)"), cst.Call)
  result: cst.CSTNode | cst.Call = transform_scheduler_init(call_node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert isinstance(result.func.value, cst.Name)
  assert result.func.value.value == "target"
  assert result.func.attr.value == "api"


def test_transform_scheduler_init_cosine_lr() -> None:
  """Transforms scheduler initialization cosine lr."""
  ctx: DummyContext = DummyContext("CosineAnnealingLR", "target.api")
  call_node: cst.Call = cst.ensure_type(
    cst.parse_expression("CosineAnnealingLR(optimizer, T_max=10, eta_min=0)"), cst.Call
  )
  result: cst.CSTNode | cst.Call = transform_scheduler_init(call_node, ctx)
  assert isinstance(result, cst.Call)
  rendered: str = cst.Module([]).code_for_node(result)
  assert rendered.startswith("target.api(")
  assert "decay_steps=10" in rendered
  assert "alpha=0" in rendered
  assert len(result.args) == 3


def test_transform_step_lr_detailed() -> None:
  """Transforms step lr detailed."""
  ctx: DummyContext = DummyContext("StepLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR()"), cst.Call)
  result = _transform_step_lr(
    call_node,
    ctx,
    "target.api",
  )
  assert len(result.args) == 2
  call_node2: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, 30, 0.1)"), cst.Call)
  result2 = _transform_step_lr(
    call_node2,
    ctx,
    "target.api",
  )
  assert len(result2.args) == 4
  kw1: str | None = result2.args[1].keyword.value if result2.args[1].keyword else None
  kw2: str | None = result2.args[2].keyword.value if result2.args[2].keyword else None
  assert kw1 == "transition_steps"
  assert kw2 == "decay_rate"
  call_node3: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, gamma=0.1, step_size=30)"), cst.Call)
  result3 = _transform_step_lr(
    call_node3,
    ctx,
    "target.api",
  )
  args_kws = [arg.keyword.value for arg in result3.args if arg.keyword]
  assert "transition_steps" in args_kws
  assert "decay_rate" in args_kws
  call_node4: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, 30)"), cst.Call)
  result4 = _transform_step_lr(
    call_node4,
    ctx,
    "target.api",
  )
  args_kws2 = [arg.keyword.value for arg in result4.args if arg.keyword]
  assert "transition_steps" in args_kws2
  assert "decay_rate" not in args_kws2
  ctx_variant: DummyContext = DummyContext(
    "StepLR",
    "target.api",
    {
      "initial_learning_rate": "custom_init",
      "step_size": "custom_step",
      "gamma": "custom_gamma",
      "staircase": "custom_stair",
    },
  )
  call_node5: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, step_size=30, gamma=0.1)"), cst.Call)
  result5 = _transform_step_lr(
    call_node5,
    ctx_variant,
    "target.api",
  )
  args_kws3 = [arg.keyword.value for arg in result5.args if arg.keyword]
  assert "custom_init" in args_kws3
  assert "custom_step" in args_kws3
  assert "custom_gamma" in args_kws3
  assert "custom_stair" in args_kws3


def test_transform_cosine_lr_detailed() -> None:
  """Transforms cosine lr detailed."""
  ctx: DummyContext = DummyContext("CosineAnnealingLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR()"), cst.Call)
  result = _transform_cosine_lr(
    call_node,
    ctx,
    "target.api",
  )
  assert len(result.args) == 1
  call_node2: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, 10, 0)"), cst.Call)
  result2 = _transform_cosine_lr(
    call_node2,
    ctx,
    "target.api",
  )
  assert len(result2.args) == 3
  kw1: str | None = result2.args[1].keyword.value if result2.args[1].keyword else None
  kw2: str | None = result2.args[2].keyword.value if result2.args[2].keyword else None
  assert kw1 == "decay_steps"
  assert kw2 == "alpha"
  call_node3: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, eta_min=0, T_max=10)"), cst.Call)
  result3 = _transform_cosine_lr(
    call_node3,
    ctx,
    "target.api",
  )
  args_kws = [arg.keyword.value for arg in result3.args if arg.keyword]
  assert "decay_steps" in args_kws
  assert "alpha" in args_kws
  call_node4: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, 10)"), cst.Call)
  result4 = _transform_cosine_lr(
    call_node4,
    ctx,
    "target.api",
  )
  assert len(result4.args) == 2
  assert result4.args[1].keyword is not None
  assert result4.args[1].keyword.value == "decay_steps"
  call_node5: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, eta_min=0)"), cst.Call)
  result5 = _transform_cosine_lr(
    call_node5,
    ctx,
    "target.api",
  )
  assert len(result5.args) == 2
  assert result5.args[1].keyword is not None
  assert result5.args[1].keyword.value == "alpha"
  ctx_variant: DummyContext = DummyContext(
    "CosineAnnealingLR",
    "target.api",
    {"initial_learning_rate": "custom_init", "T_max": "custom_T", "eta_min": "custom_eta"},
  )
  call_node6: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, T_max=10, eta_min=0)"), cst.Call)
  result6 = _transform_cosine_lr(
    call_node6,
    ctx_variant,
    "target.api",
  )
  args_kws2 = [arg.keyword.value for arg in result6.args if arg.keyword]
  assert "custom_init" in args_kws2
  assert "custom_T" in args_kws2
  assert "custom_eta" in args_kws2


def test_transform_scheduler_step() -> None:
  """Transforms scheduler step."""
  ctx: DummyContext = DummyContext("noop", "api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("scheduler.step()"), cst.Call)
  result: cst.CSTNode | cst.Call | cst.Name = transform_scheduler_step(call_node, ctx)
  assert isinstance(result, cst.Name)
  assert result.value == "None"


def test_step_lr_extra_args() -> None:
  """Tests _transform_step_lr with extra keyword and extra positional arguments."""
  ctx = DummyContext("StepLR", "optax.piecewise_constant_schedule")
  call = cst.ensure_type(
    cst.parse_expression("optim.lr_scheduler.StepLR(optimizer, 10, 0.1, 999, verbose=True)"), cst.Call
  )
  res = _transform_step_lr(
    call,
    ctx,
    "optax.piecewise_constant_schedule",
  )
  assert isinstance(res, cst.Call)
  rendered: str = cst.Module([]).code_for_node(res)
  assert rendered.startswith("optax.piecewise_constant_schedule(")
  assert "transition_steps=10" in rendered
  assert "decay_rate=0.1" in rendered
  assert len(res.args) == 4


def test_cosine_lr_extra_args() -> None:
  """Tests _transform_cosine_lr with extra keyword and extra positional arguments."""
  ctx = DummyContext("CosineAnnealingLR", "optax.cosine_decay_schedule")
  call = cst.ensure_type(
    cst.parse_expression("optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0.001, 999, verbose=True)"), cst.Call
  )
  res = _transform_cosine_lr(
    call,
    ctx,
    "optax.cosine_decay_schedule",
  )
  assert isinstance(res, cst.Call)
  rendered: str = cst.Module([]).code_for_node(res)
  assert rendered.startswith("optax.cosine_decay_schedule(")
  assert "decay_steps=50" in rendered
  assert "alpha=0.001" in rendered
  assert len(res.args) == 3
