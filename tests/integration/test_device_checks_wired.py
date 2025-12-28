"""
Integration test verifying the Device Checks plugin is now live and wired properly.
"""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
import ml_switcheroo.core.hooks as hooks
import ml_switcheroo.frameworks

# Explicitly import the plugin module so we can reload it to force hook registration
import ml_switcheroo.plugins.device_checks


@pytest.fixture
def semantics():
  # FIX: Previous tests might have cleared the global _HOOKS registry.
  # Since Python caches modules, re-running 'load_plugins' won't re-execute decorators
  # if module is already imported. We must force reload the plugin module containing
  # the 'cuda_is_available' hook.
  importlib.reload(ml_switcheroo.plugins.device_checks)
  hooks._PLUGINS_LOADED = True

  # Must ensure adaptors are loaded into registry
  importlib.reload(ml_switcheroo.frameworks)

  return SemanticsManager()


def test_cuda_check_to_jax(semantics):
  # Logic:
  # 1. Source: "if torch.cuda.is_available(): run_gpu()"
  # 2. Pivot: Rewriter identifies `torch.cuda.is_available` maps to `CudaAvailable`.
  # 3. Target (JAX): `CudaAvailable` maps to `jax.devices` + `requires_plugin=cuda_is_available`.
  # 4. Plugin: `transform_cuda_check` gets called, delegates to `JaxCoreAdapter.get_device_check_syntax()`.
  # 5. Output: `len(jax.devices('gpu')) > 0`.

  code = "if torch.cuda.is_available(): run_gpu()"

  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)

  assert result.success, f"Failed: {result.errors}"
  # The orphan plugin transforms this specific string
  assert "len(jax.devices('gpu')) > 0" in result.code
