"""Test suite for the Gradient Modes module."""

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH: str = "\nimport torch\n\ndef evaluate(model, x):\n    with torch.no_grad():\n        return model(x)\n"
EXPECTED_JAX: str = "\nimport contextlib\nimport torch\n\ndef evaluate(model, x):\n    with contextlib.nullcontext():\n        return model(x)\n"


@pytest.fixture
def manager() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for gradient modes tests.

  Returns:
      Configured `manager` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  no_grad_def: dict[str, JsonValue] = {
    "op_type": OpType.CONTEXT.value,
    "std_args": [],
    "variants": {"torch": {"api": "torch.no_grad"}, "jax": {"api": "contextlib.nullcontext"}},
  }
  mgr.data["no_grad"] = no_grad_def
  mgr._reverse_index["torch.no_grad"] = ("no_grad", no_grad_def)
  mgr._source_registry["contextlib"] = ("python", SemanticTier.EXTRAS)
  if "jax" not in mgr._providers:
    mgr._providers["jax"] = {}
  mgr._providers["jax"][SemanticTier.EXTRAS] = {"root": "contextlib", "alias": None, "sub": None}
  return mgr


def test_context_manager_rewrite(manager: SemanticsManager) -> None:
  """Verify `RuntimeConfig` handles context manager rewrite as expected.

  Args:
      manager: Populated SemanticsManager fixture.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  engine = ASTEngine(semantics=manager, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  assert result.success
  code: str = result.code
  assert "with contextlib.nullcontext():" in code
  assert "torch.no_grad" not in code
