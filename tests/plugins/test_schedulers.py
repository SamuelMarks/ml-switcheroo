"""Test suite for the Schedulers module."""

from pydantic import JsonValue

import libcst as cst

from ml_switcheroo.plugins.schedulers import (
  _create_dotted_name,
  _get_target_arg_name,
  _transform_cosine_lr,
  _transform_step_lr,
  transform_scheduler_init,
  transform_scheduler_step,
)


class DummyVariant:
  """Test helper class DummyVariant."""

  def __init__(self, args: dict[str, JsonValue] | None = None) -> None:
    """Initializes the DummyVariant instance.

    Args:
        args: The args (dict[str, JsonValue] | None) for init.
    """
    self.args = args


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(
    self,
    op_id: str | None,
    api: str | None,
    variant_args: dict[str, JsonValue] | None = None,
  ) -> None:
    """Initializes the DummyContext instance.

    Args:
        op_id: The op id (str | None) for init.
        api: The api (str | None) for init.
        variant_args: The variant args (dict[str, JsonValue] | None) for init.
    """
    self.current_op_id: str | None = op_id
    self._api: str | None = api
    self.current_variant: DummyVariant | None = (
      DummyVariant(variant_args) if variant_args is not None else DummyVariant({})
    )

  def lookup_api(self, op_id: str) -> str | None:
    """Mock implementation of lookup API.

    Args:
        op_id: The op id (str) for lookup api.

    Returns:
        Resulting `str | None` from lookup api.
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
  assert getattr(
    {
      "v": getattr({"v": _get_target_arg_name}, "__getitem__")("v")(
        getattr({"v": ctx1}, "__getitem__")("v"),
        getattr({"v": "std_name"}, "__getitem__")("v"),
        getattr({"v": "default"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "default"}, "__getitem__")("v")
  ctx2: DummyContext = DummyContext("op", "api", variant_args=None)
  if ctx2.current_variant is not None:
    ctx2.current_variant.args = None
  assert getattr(
    {
      "v": getattr({"v": _get_target_arg_name}, "__getitem__")("v")(
        getattr({"v": ctx2}, "__getitem__")("v"),
        getattr({"v": "std_name"}, "__getitem__")("v"),
        getattr({"v": "default"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "default"}, "__getitem__")("v")
  ctx3: DummyContext = DummyContext("op", "api", variant_args={})
  assert getattr(
    {
      "v": getattr({"v": _get_target_arg_name}, "__getitem__")("v")(
        getattr({"v": ctx3}, "__getitem__")("v"),
        getattr({"v": "std_name"}, "__getitem__")("v"),
        getattr({"v": "default"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "default"}, "__getitem__")("v")
  ctx4: DummyContext = DummyContext("op", "api", variant_args={"std_name": "target_name"})
  assert getattr(
    {
      "v": getattr({"v": _get_target_arg_name}, "__getitem__")("v")(
        getattr({"v": ctx4}, "__getitem__")("v"),
        getattr({"v": "std_name"}, "__getitem__")("v"),
        getattr({"v": "default"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "target_name"}, "__getitem__")("v")


def test_transform_scheduler_init_no_api() -> None:
  """Transforms scheduler initialization no API."""
  ctx: DummyContext = DummyContext("StepLR", None)
  call_node: cst.Call = cst.Call(func=cst.Name("StepLR"))
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert result is call_node


def test_transform_scheduler_init_unknown_op() -> None:
  """Transforms scheduler initialization unknown op."""
  ctx: DummyContext = DummyContext("UnknownLR", "target.api")
  call_node: cst.Call = cst.Call(func=cst.Name("UnknownLR"))
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert result is call_node


def test_transform_scheduler_init_none_op_id() -> None:
  """Transforms scheduler initialization none op id."""
  ctx: DummyContext = DummyContext(None, "target.api")
  call_node: cst.Call = cst.Call(func=cst.Name("UnknownLR"))
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert result is call_node


def test_transform_scheduler_init_step_lr() -> None:
  """Transforms scheduler initialization step lr."""
  ctx: DummyContext = DummyContext("StepLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)"), cst.Call)
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
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
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)


def test_transform_step_lr_detailed() -> None:
  """Transforms step lr detailed."""
  ctx: DummyContext = DummyContext("StepLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR()"), cst.Call)
  result = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result.args) == 2
  call_node2: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, 30, 0.1)"), cst.Call)
  result2 = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call_node2}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result2.args) == 4
  kw1: str | None = result2.args[1].keyword.value if result2.args[1].keyword else None
  kw2: str | None = result2.args[2].keyword.value if result2.args[2].keyword else None
  assert kw1 == "transition_steps"
  assert kw2 == "decay_rate"
  call_node3: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, gamma=0.1, step_size=30)"), cst.Call)
  result3 = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call_node3}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  args_kws = [arg.keyword.value for arg in result3.args if arg.keyword]
  assert "transition_steps" in args_kws
  assert "decay_rate" in args_kws
  call_node4: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optim, 30)"), cst.Call)
  result4 = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call_node4}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
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
  result5 = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call_node5}, "__getitem__")("v"),
        getattr({"v": ctx_variant}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  args_kws3 = [arg.keyword.value for arg in result5.args if arg.keyword]
  assert "custom_init" in args_kws3
  assert "custom_step" in args_kws3
  assert "custom_gamma" in args_kws3
  assert "custom_stair" in args_kws3


def test_transform_cosine_lr_detailed() -> None:
  """Transforms cosine lr detailed."""
  ctx: DummyContext = DummyContext("CosineAnnealingLR", "target.api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR()"), cst.Call)
  result = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result.args) == 1
  call_node2: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, 10, 0)"), cst.Call)
  result2 = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node2}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result2.args) == 3
  kw1: str | None = result2.args[1].keyword.value if result2.args[1].keyword else None
  kw2: str | None = result2.args[2].keyword.value if result2.args[2].keyword else None
  assert kw1 == "decay_steps"
  assert kw2 == "alpha"
  call_node3: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, eta_min=0, T_max=10)"), cst.Call)
  result3 = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node3}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  args_kws = [arg.keyword.value for arg in result3.args if arg.keyword]
  assert "decay_steps" in args_kws
  assert "alpha" in args_kws
  call_node4: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, 10)"), cst.Call)
  result4 = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node4}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result4.args) == 2
  assert result4.args[1].keyword is not None
  assert result4.args[1].keyword.value == "decay_steps"
  call_node5: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, eta_min=0)"), cst.Call)
  result5 = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node5}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert len(result5.args) == 2
  assert result5.args[1].keyword is not None
  assert result5.args[1].keyword.value == "alpha"
  ctx_variant: DummyContext = DummyContext(
    "CosineAnnealingLR",
    "target.api",
    {"initial_learning_rate": "custom_init", "T_max": "custom_T", "eta_min": "custom_eta"},
  )
  call_node6: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optim, T_max=10, eta_min=0)"), cst.Call)
  result6 = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call_node6}, "__getitem__")("v"),
        getattr({"v": ctx_variant}, "__getitem__")("v"),
        getattr({"v": "target.api"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  args_kws2 = [arg.keyword.value for arg in result6.args if arg.keyword]
  assert "custom_init" in args_kws2
  assert "custom_T" in args_kws2
  assert "custom_eta" in args_kws2


def test_transform_scheduler_step() -> None:
  """Transforms scheduler step."""
  ctx: DummyContext = DummyContext("noop", "api")
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("scheduler.step()"), cst.Call)
  result: cst.CSTNode | cst.Call | cst.Name = getattr(
    {
      "v": getattr({"v": transform_scheduler_step}, "__getitem__")("v")(
        getattr({"v": call_node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Name)
  assert result.value == "None"


def test_step_lr_extra_args() -> None:
  """Tests _transform_step_lr with extra keyword and extra positional arguments."""
  ctx = DummyContext("StepLR", "optax.piecewise_constant_schedule")
  call = cst.ensure_type(
    cst.parse_expression("optim.lr_scheduler.StepLR(optimizer, 10, 0.1, 999, verbose=True)"), cst.Call
  )
  res = getattr(
    {
      "v": getattr({"v": _transform_step_lr}, "__getitem__")("v")(
        getattr({"v": call}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "optax.piecewise_constant_schedule"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert res is not None


def test_cosine_lr_extra_args() -> None:
  """Tests _transform_cosine_lr with extra keyword and extra positional arguments."""
  ctx = DummyContext("CosineAnnealingLR", "optax.cosine_decay_schedule")
  call = cst.ensure_type(
    cst.parse_expression("optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0.001, 999, verbose=True)"), cst.Call
  )
  res = getattr(
    {
      "v": getattr({"v": _transform_cosine_lr}, "__getitem__")("v")(
        getattr({"v": call}, "__getitem__")("v"),
        getattr({"v": ctx}, "__getitem__")("v"),
        getattr({"v": "optax.cosine_decay_schedule"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert res is not None
