"""Test suite for the Roundtrip Silu module."""

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def semantics_env() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for roundtrip silu tests.

  Returns:
      Configured `semantics_env` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  silu_def: dict[str, JsonValue] = {
    "std_args": [{"name": "x", "type": "Tensor"}],
    "variants": {
      "flax_nnx": {"api": "flax.nnx.silu"},
      "torch": {"api": "torch.nn.functional.silu"},
      "numpy": {"macro_template": "{x} * (1 / (1 + np.exp(-{x})))", "required_imports": ["import numpy as np"]},
      "tensorflow": {"api": "tensorflow.nn.silu"},
    },
  }
  mgr.data["SiLU"] = silu_def
  mgr._reverse_index["flax.nnx.silu"] = ("SiLU", silu_def)
  if "flax_nnx" not in mgr.framework_configs:
    mgr.framework_configs["flax_nnx"] = {"alias": {"module": "flax.nnx", "name": "nnx"}}
  mgr._key_origins["SiLU"] = SemanticTier.ARRAY_API.value
  return mgr


def test_silu_flax_to_torch(semantics_env: SemanticsManager) -> None:
  """Verify `RuntimeConfig` handles silu flax to torch as expected.

  Args:
      semantics_env: Input `semantics_env` parameter value (`SemanticsManager`).
  """
  source: str = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result: ConversionResult = engine.run(source)
  assert result.success
  assert "F.silu(" in result.code or "torch.nn.functional.silu" in result.code


def test_silu_flax_to_numpy_macro(semantics_env: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `np.exp` for silu flax to numpy macro.

  Args:
      semantics_env: Input `semantics_env` parameter value (`SemanticsManager`).
  """
  source: str = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="numpy")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result: ConversionResult = engine.run(source)
  assert result.success
  assert "np.exp" in result.code
  assert "1 + np.exp" in result.code


def test_silu_flax_to_tensorflow(semantics_env: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `nn.silu(x)` for silu flax to tensorflow.

  Args:
      semantics_env: Input `semantics_env` parameter value (`SemanticsManager`).
  """
  source: str = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="tensorflow")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result: ConversionResult = engine.run(source)
  assert result.success
  assert "nn.silu(x)" in result.code
