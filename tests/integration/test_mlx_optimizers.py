"""
Integration Tests for Apple MLX Optimizer Transpilation.

This module verifies the translation logic between PyTorch imperative optimizers
and Apple MLX's functional/stateless optimizers.

Key Scenarios Covered:
1.  **Constructor Mapping**:
    - Maps `torch.optim.Adam` to `mlx.optimizers.Adam`.
    - Renames arguments (e.g., `lr` -> `learning_rate`).
    - Strips stateful arguments (e.g., `params`) which are handled differently in MLX.

2.  **Step Transformation**:
    - Converts `optimizer.step()` to `optimizer.update(model, grads)`.
    - **Note**: Since `model` and `grads` variables are not present in the imperative
      `step()` call, this transformation injects placeholders.

3.  **Zero Grad Removal**:
    - Converts `optimizer.zero_grad()` to `None` (No-op).
    - MLX's `value_and_grad` transform handles gradient accumulation implicitly,
      rendering manual zeroing obsolete.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.mlx_optimizers import (
  transform_mlx_optimizer_init,
  transform_mlx_optimizer_step,
  transform_mlx_zero_grad,
)

# Source Code: Standard PyTorch Training Loop Snippet
SOURCE_CODE = """
import torch.optim as optim

def setup_training(model): 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    optimizer.step() 
    optimizer.zero_grad() 
    return optimizer
"""


@pytest.fixture
def functional_framework_setup():
  """Registers a dummy framework for testing decoupling."""

  @register_framework("functional_fw")
  class FunctionalAdapter:
    pass

  return "functional_fw"


@pytest.fixture
def mlx_semantics(functional_framework_setup):
  """
  Sets up a Mock SemanticsManager wired with MLX-specific definitions
  and registered plugins.
  """
  fw_key = functional_framework_setup

  # 1. Register Plugins (simulate system bootstrap)
  _HOOKS["mlx_optimizer_init"] = transform_mlx_optimizer_init
  _HOOKS["mlx_optimizer_step"] = transform_mlx_optimizer_step
  _HOOKS["mlx_zero_grad"] = transform_mlx_zero_grad

  mgr = MagicMock(spec=SemanticsManager)

  # 2. Define Mappings
  mappings = {
    "Adam": {
      "std_args": ["params", "lr"],
      "variants": {
        "torch": {"api": "torch.optim.Adam"},
        fw_key: {
          "api": f"functional.optim.Adam",
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
    # Fix for Strict Mode: Define mapping for 'parameters' call
    # so engine allows it. The plugin will strip it anyway.
    "parameters": {
      "std_args": [],
      "variants": {"torch": {"api": "model.parameters"}, fw_key: {"api": "model.parameters", "status": "ignored"}},
    },
  }

  # 3. Configure Mock Lookups
  def get_def(name):
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
    if aid in mappings and fw == fw_key:
      return mappings[aid]["variants"][fw_key]
    return None

  mgr.get_definition.side_effect = get_def
  # mgr.get_known_apis is used by ImportFixer/Discovery
  mgr.get_known_apis.return_value = mappings
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  # Critical for avoiding "too many values to unpack" errors during import fixing
  mgr.get_import_map.return_value = {}

  return mgr


def test_mlx_optimizer_transformation(mlx_semantics, functional_framework_setup):
  """
  Verifies the end-to-end transformation of an optimizer workflow.

  Expectations:
  1. `optim.Adam` -> `functional.optim.Adam` (mock api).
  2. `lr` kwarg -> `learning_rate`.
  3. `step()` -> `update`.
  4. `zero_grad()` -> `None`.
  """
  target = functional_framework_setup
  config = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=True)
  engine = ASTEngine(semantics=mlx_semantics, config=config)

  result = engine.run(SOURCE_CODE)

  # The transformation succeeds
  assert result.success
  code = result.code

  # 1. Constructor & Argument Renaming
  # The plugin explicitly renames 'lr' to 'learning_rate'
  assert "functional.optim.Adam(learning_rate=0.001)" in code

  # 2. Step Translation checking
  # We expect the plugin to generate a placeholder update call
  assert "optimizer.update(model, grads)" in code

  # 3. Zero Grad Removal
  # Checks for direct string presence: None() or simply None expression
  assert "None" in code or "pass" in code


def test_init_transform(mlx_semantics, functional_framework_setup):
  # Setup Rewriter with context
  target = functional_framework_setup
  from ml_switcheroo.core.rewriter import PivotRewriter

  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  rewriter = PivotRewriter(mlx_semantics, cfg)

  code = "opt = torch.optim.Adam(params, lr=0.1)"
  tree = cst.parse_module(code)

  # We must set OP ID for rewriter to pass it to context
  rewriter.ctx.current_op_id = "Adam"

  res = tree.visit(rewriter).code

  # Check custom API usage
  assert "functional.optim.Adam" in res
  # Check arg rename logic
  assert "learning_rate=0.1" in res
  # Check arg strip
  assert "params" not in res


def test_step_transform():
  """
  Verify transformation logic for optimizer.step().
  Note: EscapeHatch behavior on Expressions returns the node but with markers IF it was a statement.
  On pure expression nodes without a module wrapper, it might just return the node.
  Plugin `transform_mlx_optimizer_step` is returning `EscapeHatch.mark_failure(new_call)`.
  `new_call` is `optimizer.update(...)`.
  So we check if the returned node IS that update call.
  """
  code = "opt.step()"
  node = cst.parse_expression(code)
  res = transform_mlx_optimizer_step(node, MagicMock())

  # Check content of the call - it should be 'update'
  # EscapeHatch might wrap it in FlattenSentinel or return node.
  target = res
  if isinstance(res, cst.FlattenSentinel):
    # Extract the node
    target = res.nodes[0]

  assert isinstance(target, cst.Call)
  assert target.func.attr.value == "update"
  assert len(target.args) == 2


def test_zero_grad_transform():
  code = "opt.zero_grad()"
  node = cst.parse_expression(code)
  res = transform_mlx_zero_grad(node, MagicMock())

  assert isinstance(res, cst.Name)
  assert res.value == "None"
