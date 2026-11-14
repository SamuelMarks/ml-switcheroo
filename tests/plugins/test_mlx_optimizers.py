"""Test suite for mlx optimizers."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_optimizers import (
  _create_dotted_name,
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)


def _parse_call(code: str) -> cst.Call:
  """Parse a single call expression statement into a ``cst.Call`` node.

  Args:
      code: Python source string containing a single call expression statement.

  Returns:
      Extracted ``cst.Call`` CST node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def test_create_dotted_name() -> None:
  """Verify `_create_dotted_name` produces expected results across MLX optimizers inputs."""
  node = _create_dotted_name("mlx.optimizers.Adam")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Adam"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "optimizers"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "mlx"


def test_transform_optimizer_init_with_lookup() -> None:
  """Verify `transform_mlx_optimizer_init` handles transform optimizer init with lookup as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code = "optim.Adam(model.parameters(), lr=0.01, beta1=0.9)"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2
  assert transformed.args[1].keyword is not None
  assert transformed.args[1].keyword.value == "beta1"


def test_transform_optimizer_init_fallback_attribute() -> None:
  """Verify unmapped attribute optimizer call aborts safely without hardcoded fallback."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.semantics = MagicMock()
  ctx.lookup_api.return_value = None

  code = "optim.Adam(lr=0.01)"
  call_node = _parse_call(code)

  res = transform_mlx_optimizer_init(call_node, ctx)
  assert res is call_node


def test_transform_optimizer_init_fallback_name() -> None:
  """Verify unmapped bare-name optimizer call aborts safely without hardcoded fallback."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None
  ctx.semantics = MagicMock()
  ctx.lookup_api.return_value = None

  code = "Adam(lr=0.01)"
  call_node = _parse_call(code)

  res = transform_mlx_optimizer_init(call_node, ctx)
  assert res is call_node


def test_transform_optimizer_init_fallback_other() -> None:
  """Verify scalar positional learning rate and keyword params stripping when mapped."""
  from ml_switcheroo.core.dsl import FrameworkVariant

  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  ctx.current_variant = FrameworkVariant(api="mlx.optimizers.Adam", args={"wd": "weight_decay"})

  code = "Adam(0.01, params=p, wd=0.001)"
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  call_node = cst.ensure_type(expr.value, cst.Call)

  res = transform_mlx_optimizer_init(call_node, ctx)
  rendered = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
  assert "mlx.optimizers.Adam(0.01" in rendered
  assert "weight_decay=0.001" in rendered
  assert "params=" not in rendered

  neg_call = cst.ensure_type(cst.parse_expression("Adam(-0.01)"), cst.Call)
  neg_res = transform_mlx_optimizer_init(neg_call, ctx)
  assert len(neg_res.args) == 1


def test_transform_optimizer_step_attribute() -> None:
  """Verify `transform_mlx_optimizer_step` handles transform optimizer step attribute as expected."""
  ctx = MagicMock(spec=HookContext)

  code = "my_opt.step()"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "my_opt"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_optimizer_step_name() -> None:
  """Verify `transform_mlx_optimizer_step` returns `2` for transform optimizer step name."""
  ctx = MagicMock(spec=HookContext)

  code = "step()"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_step(call_node, ctx)

  assert isinstance(transformed, cst.Call)

  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "optimizer"
  assert transformed.func.attr.value == "update"
  assert len(transformed.args) == 2


def test_transform_zero_grad() -> None:
  """Verify `transform_mlx_zero_grad` handles transform zero grad as expected."""
  ctx = MagicMock(spec=HookContext)

  code = "opt.zero_grad()"
  call_node = _parse_call(code)

  transformed = transform_mlx_zero_grad(call_node, ctx)

  assert isinstance(transformed, cst.Name)
  assert transformed.value == "None"


def test_transform_optimizer_init_positional_args() -> None:
  """Verify `transform_mlx_optimizer_init` handles transform optimizer init positional args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.AdamW"

  code = "optim.Adam(model.parameters(), 0.01, 0.9)"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)

  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "AdamW"
  assert len(transformed.args) == 2


def test_transform_optimizer_init_kwarg_no_value() -> None:
  """Verify star-kwargs are stripped when transforming mapped optimizer init."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"

  code = "optim.Adam(**kwargs)"
  call_node = _parse_call(code)

  res = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0


def test_transform_optimizer_init_lr_rename_no_keyword() -> None:
  """Verify transform optimizer init lr rename no keyword works as expected."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam(model.parameters(), lr=0.01)"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_no_args() -> None:
  """Verify `transform_mlx_optimizer_init` returns `0` for transform optimizer init no args."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam()"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert len(transformed.args) == 0


def test_transform_optimizer_init_lr_rename_no_keyword_true() -> None:
  """Verify transform optimizer init lr rename no keyword true works as expected."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam(lr=0.01)"  # no positional args
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_false() -> None:
  """Verify transform optimizer init lr rename no keyword false works as expected."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam(model.parameters(), 0.01)"  # no positional args
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert isinstance(transformed.args[0].value, cst.Float)
  assert transformed.args[0].value.value == "0.01"


def test_transform_optimizer_init_lr_rename_no_keyword_true_already() -> None:
  """Verify transform optimizer init lr rename no keyword true already works as expected."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam(learning_rate=0.01)"
  call_node = _parse_call(code)

  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"


def test_transform_optimizer_init_lr_rename_no_keyword_true_actually() -> None:
  """Verify transform optimizer init lr rename no keyword true actually works as expected."""
  ctx = MagicMock()
  ctx.current_op_id = "Adam"
  ctx.lookup_api.return_value = "mlx.optimizers.Adam"
  code = "optim.Adam(lr=0.01)"
  call_node = _parse_call(code)
  transformed = transform_mlx_optimizer_init(call_node, ctx)
  assert transformed.args[0].keyword is not None
  assert transformed.args[0].keyword.value == "learning_rate"
