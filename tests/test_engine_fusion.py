from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
import textwrap


def test_engine_fusion_jax():
  code = textwrap.dedent("""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x
    """)
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
