import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


def test_engine_sharding_torch():
  config = RuntimeConfig(source_framework="jax", target_framework="torch", enable_sharding=True)
  engine = ASTEngine(semantics=SemanticsManager(), config=config)
  res = engine.run("import jax.numpy as jnp\njnp.array([1])")
  assert "import torch" in res.code
