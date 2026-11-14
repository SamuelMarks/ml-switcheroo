"""Test suite for the Mlx Optimizers module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import register_hook
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter

SOURCE_CODE: str = "\nimport torch.optim as optim\n\ndef setup_training(model):\n    optimizer = optim.Adam(model.parameters(), lr=0.001)\n    optimizer.step()\n    optimizer.zero_grad()\n    return optimizer\n"


@pytest.fixture
def functional_framework_setup() -> str:
  """Create a configured `str` fixture for MLX optimizers tests.

  Returns:
      Configured `functional_framework_setup` fixture value (str).
  """

  @register_framework("functional_fw")
  class FunctionalAdapter:
    """Sentinel `FunctionalAdapter` fixture class used in MLX optimizers tests."""

  return "functional_fw"


@pytest.fixture
def mlx_semantics(functional_framework_setup: str) -> MagicMock:
  """Create a configured `MagicMock` fixture for MLX optimizers tests.

  Args:
      functional_framework_setup: Input `functional_framework_setup` parameter value (`str`).

  Returns:
      Configured `mlx_semantics` fixture value (MagicMock).
  """
  fw_key = functional_framework_setup
  register_hook("mlx_optimizer_init")(transform_mlx_optimizer_init)
  register_hook("mlx_optimizer_step")(transform_mlx_optimizer_step)
  register_hook("mlx_zero_grad")(transform_mlx_zero_grad)
  mgr = MagicMock(spec=SemanticsManager)
  mappings: dict[str, dict[str, JsonValue]] = {
    "Adam": {
      "std_args": ["params", "lr"],
      "variants": {
        "torch": {"api": "torch.optim.Adam"},
        fw_key: {
          "api": "functional.optim.Adam",
          "args": {"lr": "learning_rate"},
          "requires_plugin": "mlx_optimizer_init",
        },
      },
    },
    "step": {
      "std_args": [],
      "variants": {"torch": {"api": "optimizer.step"}, fw_key: {"requires_plugin": "mlx_optimizer_step"}},
    },
    "zero_grad": {
      "std_args": [],
      "variants": {"torch": {"api": "optimizer.zero_grad"}, fw_key: {"requires_plugin": "mlx_zero_grad"}},
    },
    "parameters": {
      "std_args": [],
      "variants": {"torch": {"api": "model.parameters"}, fw_key: {"api": "model.parameters", "status": "ignored"}},
    },
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "Adam" in name:
      _ret_v_1: tuple[str, dict[str, JsonValue]] | None = (
        "Adam",
        mappings["Adam"],
      )
      return _ret_v_1
    if "step" in name:
      _ret_v_2: tuple[str, dict[str, JsonValue]] | None = (
        "step",
        mappings["step"],
      )
      return _ret_v_2
    if "zero_grad" in name:
      _ret_v_3: tuple[str, dict[str, JsonValue]] | None = (
        "zero_grad",
        mappings["zero_grad"],
      )
      return _ret_v_3
    if "parameters" in name:
      _ret_v_4: tuple[str, dict[str, JsonValue]] | None = (
        "parameters",
        mappings["parameters"],
      )
      return _ret_v_4
    return ("Generic", {"variants": {}})

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in MLX optimizers tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if aid in mappings and fw == fw_key:
      variants = mappings[aid].get("variants")
      if isinstance(variants, dict):
        variant = variants.get(fw_key)
        if isinstance(variant, dict):
          return variant
    return None

  mgr.get_definition.side_effect = get_def
  mgr.get_known_apis.return_value = mappings
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  return mgr


def test_mlx_optimizer_transformation(mlx_semantics: MagicMock, functional_framework_setup: str) -> None:
  """Verify `RuntimeConfig` handles MLX optimizer transformation as expected.

  Args:
      mlx_semantics: Input `mlx_semantics` parameter value (`MagicMock`).
      functional_framework_setup: Input `functional_framework_setup` parameter value (`str`).
  """
  target: str = functional_framework_setup
  config = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=True)
  engine = ASTEngine(semantics=mlx_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_CODE)
  assert result.success
  code: str = result.code
  assert "functional.optim.Adam(learning_rate=0.001)" in code
  assert "optimizer.update(model, grads)" in code
  assert "None" in code or "pass" in code


def test_init_transform(mlx_semantics: MagicMock, functional_framework_setup: str) -> None:
  """Verify `RuntimeConfig` emits `functional.optim.Adam` for init transform.

  Args:
      mlx_semantics: Input `mlx_semantics` parameter value (`MagicMock`).
      functional_framework_setup: Input `functional_framework_setup` parameter value (`str`).
  """
  target: str = functional_framework_setup
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  rewriter = PivotRewriter(mlx_semantics, cfg)
  rewriter.context.hook_context.current_op_id = "Adam"
  code: str = "opt = torch.optim.Adam(params, lr=0.1)"
  tree = cst.parse_module(code)
  res: str = str(rewriter.convert(tree).code)
  assert "functional.optim.Adam" in res
  assert "learning_rate=0.1" in res
  assert "params" not in res


def test_step_transform() -> None:
  """Verify `transform_mlx_optimizer_step` handles step transform as expected."""
  code: str = "opt.step()"
  node = cst.parse_expression(code)
  res = transform_mlx_optimizer_step(cst.ensure_type(node, cst.Call), MagicMock())
  target: cst.CSTNode
  if isinstance(res, cst.FlattenSentinel):
    target = res.nodes[0]
  elif isinstance(res, cst.RemovalSentinel):
    pytest.fail("Unexpected RemovalSentinel")
  else:
    target = res

  assert "update" in cst.Module([]).code_for_node(target)


def test_zero_grad_transform() -> None:
  """Verify `transform_mlx_zero_grad` handles zero grad transform as expected."""
  code: str = "opt.zero_grad()"
  node = cst.parse_expression(code)
  res = transform_mlx_zero_grad(cst.ensure_type(node, cst.Call), MagicMock())
  assert isinstance(res, cst.Name)
  assert res.value == "None"
