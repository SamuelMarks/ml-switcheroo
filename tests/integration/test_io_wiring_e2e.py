"""Test suite for the Io Wiring E2E module."""

import pytest
import typing
import importlib
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE: str = "\nimport torch\n\ndef save_model(model):\n    torch.save(model, 'checkpoint.pth')\n    loaded = torch.load('checkpoint.pth')\n    return loaded\n"


@pytest.fixture(autouse=True)
def ensure_io_plugin() -> typing.Generator[None, None, None]:
  """Helper to ensure I/O plugin."""
  import ml_switcheroo.core.hooks as hooks
  import ml_switcheroo.plugins.io_handler

  importlib.reload(ml_switcheroo.plugins.io_handler)
  hooks._PLUGINS_LOADED = True  # type: ignore
  yield


@pytest.fixture(scope="module")
def hydrated_semantics() -> SemanticsManager:
  """Helper to hydrated semantics."""
  mgr = SemanticsManager()
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


def test_io_to_jax(hydrated_semantics: SemanticsManager) -> None:
  """Verifies the behavior of I/O to JAX."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_CODE)
  assert result.success, f"Errors: {result.errors}"
  code: str = result.code
  assert "import orbax.checkpoint" in code
  assert "PyTreeCheckpointer().save" in code
  assert "directory='checkpoint.pth'" in code
  assert "item=model" in code
  assert "PyTreeCheckpointer().restore" in code


def test_io_to_numpy(hydrated_semantics: SemanticsManager) -> None:
  """Verifies the behavior of I/O to NumPy."""
  config = RuntimeConfig(source_framework="torch", target_framework="numpy", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_CODE)
  assert result.success, f"Errors: {result.errors}"
  code: str = result.code
  assert "np.save" in code, "Failed to map to numpy.save"
  assert "file='checkpoint.pth'" in code
  assert "arr=model" in code
  assert "np.load" in code
