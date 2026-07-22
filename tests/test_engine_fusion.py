"""Test suite for the Engine Fusion module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
import textwrap


def test_engine_fusion_jax():
  """Verifies the behavior of engine fusion JAX."""
  code = textwrap.dedent(
    "\n        class Model(nn.Module):\n            def __init__(self):\n                super().__init__()\n            def forward(self, x):\n                return x\n    "
  )
  config = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    target_flavour="linen",
    enable_sharding=True,
    enable_graph_optimizer=True,
  )
  sm = SemanticsManager()
  engine = ASTEngine(sm, config)
  res = engine.run(code)
  assert res.success
