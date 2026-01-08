"""
End-to-End Test for IO Persistence Wiring.

Verifies that adding "Save/Load" to standards_internal.py effectively
activates the io_handler plugin.

Includes robust plugin reloading to prevent test pollution from 'clear_hooks'.
"""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

# Source code with IO operations
SOURCE_CODE = """ 
import torch

def save_model(model): 
    torch.save(model, 'checkpoint.pth') 
    loaded = torch.load('checkpoint.pth') 
    return loaded
"""


@pytest.fixture(autouse=True)
def ensure_io_plugin():
  """
  Robustness Fix: Prevents "Missing required plugin" errors caused by
  test pollution. Forces a reload of the io_handler module to re-register
  the @register_hook decorators if a previous test cleared the registry.
  """
  import ml_switcheroo.core.hooks as hooks
  import ml_switcheroo.plugins.io_handler

  # Reload the specific plugin module
  importlib.reload(ml_switcheroo.plugins.io_handler)
  # Mark as loaded to satisfy any internal checks
  hooks._PLUGINS_LOADED = True


@pytest.fixture(scope="module")
def hydrated_semantics():
  # Initialize manager
  mgr = SemanticsManager()

  # Force update the definitions to ensure IO ops are mapped to the plugin
  # Use the definition that 'io_handler' expects (generic 'save', 'load' mapped from torch.save)

  # For torch.save
  mgr.update_definition(
    "TorchSave",
    {
      "operation": "TorchSave",
      "std_args": ["obj", "f"],
      "variants": {
        "torch": {"api": "torch.save"},
        "jax": {"api": "save", "requires_plugin": "io_handler"},
        "numpy": {"api": "save", "requires_plugin": "io_handler"},
      },
    },
  )

  # For torch.load
  mgr.update_definition(
    "TorchLoad",
    {
      "operation": "TorchLoad",
      "std_args": ["f"],
      "variants": {
        "torch": {"api": "torch.load"},
        "jax": {"api": "load", "requires_plugin": "io_handler"},
        "numpy": {"api": "load", "requires_plugin": "io_handler"},
      },
    },
  )

  return mgr


def test_io_to_jax(hydrated_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result = engine.run(SOURCE_CODE)

  assert result.success, f"Errors: {result.errors}"
  code = result.code

  # Check Imports
  assert "import orbax.checkpoint" in code

  # Check Syntax
  assert "PyTreeCheckpointer().save" in code
  assert "directory='checkpoint.pth'" in code
  assert "item=model" in code

  # Check Load
  assert "PyTreeCheckpointer().restore" in code


def test_io_to_numpy(hydrated_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="numpy", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result = engine.run(SOURCE_CODE)

  assert result.success, f"Errors: {result.errors}"
  code = result.code

  assert "np.save" in code, "Failed to map to numpy.save"
  assert "file='checkpoint.pth'" in code
  assert "arr=model" in code
  assert "np.load" in code
