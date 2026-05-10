from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from unittest.mock import MagicMock
import textwrap


def test_engine_fusion_gap():
  code = textwrap.dedent("""
        import torch.nn as nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
            def forward(self, x):
                return self.fc(x)
    """)
  config = RuntimeConfig(
    source_framework="torch", target_framework="jax", enable_sharding=True, enable_graph_optimization=True
  )
  sm = SemanticsManager()
  engine = ASTEngine(sm, config)

  # Bypass the top-level dispatch to force testing the loopback logic in _run_rewriter_pipeline
  res = engine._run_rewriter_pipeline(code, MagicMock())
  assert res is not None
