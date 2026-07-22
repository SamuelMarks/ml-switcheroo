"""Test suite for the Bonsai E2E module."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
import pytest


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  return SemanticsManager()


QWEN_SOURCE = "import torch\nimport torch.nn as nn\n\nclass QwenBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.q_proj = nn.Linear(1024, 1024)\n        self.k_proj = nn.Linear(1024, 1024)\n        self.v_proj = nn.Linear(1024, 1024)\n\n        self.gate_proj = nn.Linear(1024, 4096)\n        self.up_proj = nn.Linear(1024, 4096)\n        self.down_proj = nn.Linear(4096, 1024)\n\n    def forward(self, x):\n        q = self.q_proj(x)\n        k = self.k_proj(x)\n        v = self.v_proj(x)\n\n        gate = self.gate_proj(x)\n        up = self.up_proj(x)\n        mlp_out = self.down_proj(gate * up)\n\n        return q, mlp_out\n"
QWEN_VL_SOURCE = "import torch\nimport torch.nn as nn\n\nclass VisionFrontEnd(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.patch_conv = nn.Conv3d(\n            in_channels=3,\n            out_channels=1280,\n            kernel_size=(2, 14, 14),\n            stride=(2, 14, 14),\n            bias=False\n        )\n\n    def forward(self, x):\n        return self.patch_conv(x)\n"


def test_qwen_to_flax_nnx(semantics):
  """Verifies the behavior of qwen to Flax NNX."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="flax_nnx", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_SOURCE)
  assert result.success
  code = result.code
  assert "qkv_proj" in code.lower()
  assert "swiglu" in code.lower()
  assert "jax.lax.with_sharding_constraint" in code
  assert "PartitionSpec" in code


def test_qwen_vl_to_mlx(semantics):
  """Verifies the behavior of qwen vl to MLX."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="mlx", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_VL_SOURCE)
  assert result.success
  code = result.code
  assert "import mlx.core as mx" in code
  assert "import mlx.nn as nn" in code
  assert "nn.Conv" in code
  assert "mx.distributed.shard" in code


def test_qwen_to_keras(semantics):
  """Verifies the behavior of qwen to Keras."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="keras", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_SOURCE)
  assert result.success
  code = result.code
  assert "keras.distribution.layout" in code
