"""
End-to-End Tests for jax-ml/bonsai architectures (Qwen3 & Qwen3-VL).

Validates the full Engine Pipeline including:
1. Sharding Metadata Extraction.
2. Qwen-specific Graph Fusions (SwiGLU, VisionPatch).
3. Multi-Framework Code Generation.
"""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
import pytest


@pytest.fixture
def semantics():
  """Function docstring."""
  return SemanticsManager()


QWEN_SOURCE = """import torch
import torch.nn as nn

class QwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)
        
        self.gate_proj = nn.Linear(1024, 4096)
        self.up_proj = nn.Linear(1024, 4096)
        self.down_proj = nn.Linear(4096, 1024)
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up) 
        
        return q, mlp_out
"""

QWEN_VL_SOURCE = """import torch
import torch.nn as nn

class VisionFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_conv = nn.Conv3d(
            in_channels=3,
            out_channels=1280,
            kernel_size=(2, 14, 14),
            stride=(2, 14, 14),
            bias=False
        )
        
    def forward(self, x):
        return self.patch_conv(x)
"""


def test_qwen_to_flax_nnx(semantics):
  """Test standard Qwen block translates to fused Flax NNX with sharding."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="flax_nnx", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_SOURCE)

  assert result.success
  code = result.code

  # Assert Fusions occurred
  assert "qkv_proj" in code.lower()
  assert "swiglu" in code.lower()

  # Assert Sharding constraints were injected
  assert "jax.lax.with_sharding_constraint" in code
  assert "PartitionSpec" in code


def test_qwen_vl_to_mlx(semantics):
  """Test Qwen VL vision front end translates to Apple MLX."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="mlx", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_VL_SOURCE)

  assert result.success
  code = result.code

  # Assert MLX semantics
  assert "import mlx.core as mx" in code
  assert "import mlx.nn as nn" in code

  # Fallback checking
  assert "nn.Conv" in code

  # Sharding
  assert "mx.distributed.shard" in code


def test_qwen_to_keras(semantics):
  """Test standard Qwen block translates to Keras 3 with layout tracking."""
  config = RuntimeConfig(
    source_framework="torch", target_framework="keras", enable_sharding=True, enable_graph_optimization=True
  )
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(QWEN_SOURCE)

  assert result.success
  code = result.code

  # Keras specific distribution comments
  assert "keras.distribution.layout" in code
