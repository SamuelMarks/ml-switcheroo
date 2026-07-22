"""Test suite for the Schedulers module."""

import libcst as cst
from ml_switcheroo.plugins.schedulers import (
  _create_dotted_name,
  _get_target_arg_name,
  transform_scheduler_init,
  _transform_step_lr,
  _transform_cosine_lr,
  transform_scheduler_step,
)


class DummyVariant:
  """Dummy Variant class for testing purposes."""

  def __init__(self, args=None):
    """Initializes the DummyVariant instance."""
    self.args = args


class DummyContext:
  """Dummy Context class for testing purposes."""

  def __init__(self, op_id, api, variant_args=None):
    """Initializes the DummyContext instance."""
    self.current_op_id = op_id
    self._api = api
    self.current_variant = DummyVariant(variant_args) if variant_args is not None else DummyVariant({})

  def lookup_api(self, op_id):
    """Mock implementation of lookup API."""
    return self._api


def test_create_dotted_name():
  """Creates dotted name."""
  node = _create_dotted_name("a")
  assert isinstance(node, cst.Name)
  assert node.value == "a"
  node = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert node.value.attr.value == "b"
  assert node.value.value.value == "a"


def test_get_target_arg_name():
  """Gets target argument name."""
  ctx1 = DummyContext("op", "api")
  ctx1.current_variant = None
  assert _get_target_arg_name(ctx1, "std_name", "default") == "default"
  ctx2 = DummyContext("op", "api", variant_args=None)
  ctx2.current_variant.args = None
  assert _get_target_arg_name(ctx2, "std_name", "default") == "default"
  ctx3 = DummyContext("op", "api", variant_args={})
  assert _get_target_arg_name(ctx3, "std_name", "default") == "default"
  ctx4 = DummyContext("op", "api", variant_args={"std_name": "target_name"})
  assert _get_target_arg_name(ctx4, "std_name", "default") == "target_name"


def test_transform_scheduler_init_no_api():
  """Transforms scheduler initialization no API."""
  ctx = DummyContext("StepLR", None)
  call_node = cst.Call(func=cst.Name("StepLR"))
  result = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_unknown_op():
  """Transforms scheduler initialization unknown op."""
  ctx = DummyContext("UnknownLR", "target.api")
  call_node = cst.Call(func=cst.Name("UnknownLR"))
  result = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_none_op_id():
  """Transforms scheduler initialization none op id."""
  ctx = DummyContext(None, "target.api")
  call_node = cst.Call(func=cst.Name("UnknownLR"))
  result = transform_scheduler_init(call_node, ctx)
  assert result is call_node


def test_transform_scheduler_init_step_lr():
  """Transforms scheduler initialization step lr."""
  ctx = DummyContext("StepLR", "target.api")
  call_node = cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)")
  result = transform_scheduler_init(call_node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.value.value == "target"
  assert result.func.attr.value == "api"


def test_transform_scheduler_init_cosine_lr():
  """Transforms scheduler initialization cosine lr."""
  ctx = DummyContext("CosineAnnealingLR", "target.api")
  call_node = cst.parse_expression("CosineAnnealingLR(optimizer, T_max=10, eta_min=0)")
  result = transform_scheduler_init(call_node, ctx)
  assert isinstance(result, cst.Call)


def test_transform_step_lr_detailed():
  """Transforms step lr detailed."""
  ctx = DummyContext("StepLR", "target.api")
  call_node = cst.parse_expression("StepLR()")
  result = _transform_step_lr(call_node, ctx, "target.api")
  assert len(result.args) == 2
  call_node = cst.parse_expression("StepLR(optim, 30, 0.1)")
  result = _transform_step_lr(call_node, ctx, "target.api")
  assert len(result.args) == 4
  kw1 = result.args[1].keyword.value
  kw2 = result.args[2].keyword.value
  assert kw1 == "transition_steps"
  assert kw2 == "decay_rate"
  call_node = cst.parse_expression("StepLR(optim, gamma=0.1, step_size=30)")
  result = _transform_step_lr(call_node, ctx, "target.api")
  args_kws = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "transition_steps" in args_kws
  assert "decay_rate" in args_kws
  call_node = cst.parse_expression("StepLR(optim, 30)")
  result = _transform_step_lr(call_node, ctx, "target.api")
  args_kws = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "transition_steps" in args_kws
  assert "decay_rate" not in args_kws
  ctx_variant = DummyContext(
    "StepLR",
    "target.api",
    {
      "initial_learning_rate": "custom_init",
      "step_size": "custom_step",
      "gamma": "custom_gamma",
      "staircase": "custom_stair",
    },
  )
  call_node = cst.parse_expression("StepLR(optim, step_size=30, gamma=0.1)")
  result = _transform_step_lr(call_node, ctx_variant, "target.api")
  args_kws = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "custom_init" in args_kws
  assert "custom_step" in args_kws
  assert "custom_gamma" in args_kws
  assert "custom_stair" in args_kws


def test_transform_cosine_lr_detailed():
  """Transforms cosine lr detailed."""
  ctx = DummyContext("CosineAnnealingLR", "target.api")
  call_node = cst.parse_expression("CosineAnnealingLR()")
  result = _transform_cosine_lr(call_node, ctx, "target.api")
  assert len(result.args) == 1
  call_node = cst.parse_expression("CosineAnnealingLR(optim, 10, 0)")
  result = _transform_cosine_lr(call_node, ctx, "target.api")
  assert len(result.args) == 3
  kw1 = result.args[1].keyword.value
  kw2 = result.args[2].keyword.value
  assert kw1 == "decay_steps"
  assert kw2 == "alpha"
  call_node = cst.parse_expression("CosineAnnealingLR(optim, eta_min=0, T_max=10)")
  result = _transform_cosine_lr(call_node, ctx, "target.api")
  args_kws = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "decay_steps" in args_kws
  assert "alpha" in args_kws
  call_node = cst.parse_expression("CosineAnnealingLR(optim, 10)")
  result = _transform_cosine_lr(call_node, ctx, "target.api")
  assert len(result.args) == 2
  assert result.args[1].keyword.value == "decay_steps"
  call_node = cst.parse_expression("CosineAnnealingLR(optim, eta_min=0)")
  result = _transform_cosine_lr(call_node, ctx, "target.api")
  assert len(result.args) == 2
  assert result.args[1].keyword.value == "alpha"
  ctx_variant = DummyContext(
    "CosineAnnealingLR",
    "target.api",
    {"initial_learning_rate": "custom_init", "T_max": "custom_T", "eta_min": "custom_eta"},
  )
  call_node = cst.parse_expression("CosineAnnealingLR(optim, T_max=10, eta_min=0)")
  result = _transform_cosine_lr(call_node, ctx_variant, "target.api")
  args_kws = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "custom_init" in args_kws
  assert "custom_T" in args_kws
  assert "custom_eta" in args_kws


def test_transform_scheduler_step():
  """Transforms scheduler step."""
  ctx = DummyContext("noop", "api")
  call_node = cst.parse_expression("scheduler.step()")
  result = transform_scheduler_step(call_node, ctx)
  assert isinstance(result, cst.Name)
  assert result.value == "None"
