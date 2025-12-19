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
from unittest.mock import MagicMock

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import _HOOKS
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
def mlx_semantics():
  """
  Sets up a Mock SemanticsManager wired with MLX-specific definitions
  and registered plugins.
  """
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
        "mlx": {"api": "mlx.optimizers.Adam", "args": {"lr": "learning_rate"}, "requires_plugin": "mlx_optimizer_init"},
      },
    },
    "step": {
      "std_args": [],
      "variants": {"torch": {"api": "optimizer.step"}, "mlx": {"requires_plugin": "mlx_optimizer_step"}},
    },
    "zero_grad": {
      "std_args": [],
      "variants": {"torch": {"api": "optimizer.zero_grad"}, "mlx": {"requires_plugin": "mlx_zero_grad"}},
    },
    # Fix for Strict Mode: Define mapping for 'parameters' call
    # so engine allows it. The plugin will strip it anyway.
    "parameters": {
      "std_args": [],
      "variants": {"torch": {"api": "model.parameters"}, "mlx": {"api": "model.parameters", "status": "ignored"}},
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
    if aid in mappings and fw == "mlx":
      return mappings[aid]["variants"]["mlx"]
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


def test_mlx_optimizer_transformation(mlx_semantics):
  """
  Verifies the end-to-end transformation of an optimizer workflow.

  Expectations:
  1. `optim.Adam` -> `mlx.optimizers.Adam`.
  2. `lr` kwarg -> `learning_rate`.
  3. `step()` -> `update`.
  4. `zero_grad()` -> `None`.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="mlx", strict_mode=True)
  engine = ASTEngine(semantics=mlx_semantics, config=config)

  result = engine.run(SOURCE_CODE)

  # The transformation succeeds
  assert result.success
  code = result.code

  # 1. Constructor & Argument Renaming
  # The plugin explicitly renames 'lr' to 'learning_rate'
  assert "mlx.optimizers.Adam(learning_rate=0.001)" in code

  # 2. Step Translation checking
  # We expect the plugin to generate a placeholder update call
  assert "optimizer.update(model, grads)" in code

  # 3. Zero Grad Removal
  # Checks for direct string presence: None() or simply None expression
  assert "None" in code or "pass" in code
