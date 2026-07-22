"""Test suite for the Mlx Optimizers module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)
from tests.conftest import TestRewriter as PivotRewriter

SOURCE_CODE = "\nimport torch.optim as optim\n\ndef setup_training(model):\n    optimizer = optim.Adam(model.parameters(), lr=0.001)\n    optimizer.step()\n    optimizer.zero_grad()\n    return optimizer\n"


@pytest.fixture
def functional_framework_setup():
  """Provides a mock functional framework setup for testing."""

  @register_framework("functional_fw")
  class FunctionalAdapter:
    """Test suite for the Functional Adapter component."""

    pass

  return "functional_fw"


@pytest.fixture
def mlx_semantics(functional_framework_setup):
  """Provides a mock MLX semantics for testing."""
  fw_key = functional_framework_setup
  _HOOKS["mlx_optimizer_init"] = transform_mlx_optimizer_init
  _HOOKS["mlx_optimizer_step"] = transform_mlx_optimizer_step
  _HOOKS["mlx_zero_grad"] = transform_mlx_zero_grad
  mgr = MagicMock(spec=SemanticsManager)
  mappings = {
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

  def get_def(name):
    """Gets def."""
    if "Adam" in name:
      return ("Adam", mappings["Adam"])
    if "step" in name:
      return ("step", mappings["step"])
    if "zero_grad" in name:
      return ("zero_grad", mappings["zero_grad"])
    if "parameters" in name:
      return ("parameters", mappings["parameters"])
    return ("Generic", {"variants": {}})

  def resolve(aid, fw):
    """Resolves ."""
    if aid in mappings and fw == fw_key:
      return mappings[aid]["variants"][fw_key]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.get_known_apis.return_value = mappings
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  mgr.get_import_map.return_value = {}
  return mgr


def test_mlx_optimizer_transformation(mlx_semantics, functional_framework_setup):
  """Verifies the behavior of MLX optimizer transformation."""
  target = functional_framework_setup
  config = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=True)
  engine = ASTEngine(semantics=mlx_semantics, config=config)
  result = engine.run(SOURCE_CODE)
  assert result.success
  code = result.code
  assert "functional.optim.Adam(learning_rate=0.001)" in code
  assert "optimizer.update(model, grads)" in code
  assert "None" in code or "pass" in code


def test_init_transform(mlx_semantics, functional_framework_setup):
  """Verifies the behavior of initialization transform."""
  target = functional_framework_setup
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  rewriter = PivotRewriter(mlx_semantics, cfg)
  rewriter.context.hook_context.current_op_id = "Adam"
  code = "opt = torch.optim.Adam(params, lr=0.1)"
  tree = cst.parse_module(code)
  res = rewriter.convert(tree).code
  assert "functional.optim.Adam" in res
  assert "learning_rate=0.1" in res
  assert "params" not in res


def test_step_transform():
  """Verifies the behavior of step transform."""
  code = "opt.step()"
  node = cst.parse_expression(code)
  res = transform_mlx_optimizer_step(node, MagicMock())
  target = res
  if isinstance(res, cst.FlattenSentinel):
    target = res.nodes[0]
  if hasattr(target, "value"):
    pass
  import libcst

  assert "update" in libcst.Module([]).code_for_node(target)


def test_zero_grad_transform():
  """Verifies the behavior of zero grad transform."""
  code = "opt.zero_grad()"
  node = cst.parse_expression(code)
  res = transform_mlx_zero_grad(node, MagicMock())
  assert isinstance(res, cst.Name)
  assert res.value == "None"
