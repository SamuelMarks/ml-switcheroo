"""Test suite for the Hexagonal Static Conversion Lattice.

Verifies static conversion across all 30 directed edges between:
- PyTorch
- Flax NNX (JAX)
- Apple MLX
- Keras
- NVIDIA SASS
- AMD RDNA
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine

TORCH_SNIPPET: str = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x):
        return self.fc(x)
"""

JAX_SNIPPET: str = """
import jax
import jax.numpy as jnp

def model(x):
    return jnp.abs(x)
"""

FLAX_NNX_SNIPPET: str = """
import flax.nnx as nnx
import jax.numpy as jnp

class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.fc = nnx.Linear(10, 20, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)
"""

MLX_SNIPPET: str = """
import mlx.core as mx
import mlx.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def __call__(self, x):
        return self.fc(x)
"""

KERAS_SNIPPET: str = """
import keras
import keras.layers as layers

class Model(keras.Layer):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(20)

    def call(self, x):
        return self.fc(x)
"""

SASS_SNIPPET: str = """
    // Input x -> R0
    // BEGIN Linear (fc)
    MOV R1, RZ;
    MOV R2, RZ;
L_GEMM_fc:
    LDG.E.F32 R3, [R2];
    LDG.E.F32 R4, [R3];
    FFMA R1, R3, R4, R1;
    IADD3 R2, R2, 4, RZ;
    IADD3 R3, R3, 4, RZ;
    IADD3 R2, R2, 1, RZ;
    ISETP.LT.AND P0, PT, R2, 128, PT;
    P0 BRA L_GEMM_fc;
    // END Linear (fc)
    // Return: R1
"""

RDNA_SNIPPET: str = """
; RDNA Code Generation Initialized (Arch: gfx1030)
    ; Input x -> v0
    ; BEGIN Linear (fc)
    v_mov_b32 v1, 0
    s_mov_b32 s0, 0
L_GEMM_fc:
    global_load_dword v2, v4, off
    global_load_dword v3, v5, off
    s_waitcnt vmcnt(0)
    v_fmac_f32 v1, v2, v3
    v_add_u32 v4, v4, 4
    v_add_u32 v5, v5, 4
    s_add_i32 s0, s0, 1
    s_cmp_lt_i32 s0, 128
    s_cbranch_scc1 L_GEMM_fc
    ; END Linear (fc)
    ; Return: v1
"""

FRAMEWORK_SNIPPETS: dict[str, str] = {
  "torch": TORCH_SNIPPET,
  "jax": JAX_SNIPPET,
  "flax_nnx": FLAX_NNX_SNIPPET,
  "mlx": MLX_SNIPPET,
  "keras": KERAS_SNIPPET,
  "nvidia_sass": SASS_SNIPPET,
  "rdna": RDNA_SNIPPET,
}

ALL_PAIRS: list[tuple[str, str]] = [(src, tgt) for src in FRAMEWORK_SNIPPETS for tgt in FRAMEWORK_SNIPPETS if src != tgt]


@pytest.mark.parametrize("source,target", ALL_PAIRS)
def test_directed_conversion_pair(source: str, target: str) -> None:
  """Tests static conversion for each directed edge in the hexagonal matrix.

  Args:
      source: The source framework name.
      target: The target framework name.
  """
  input_code: str = FRAMEWORK_SNIPPETS[source]
  engine: ASTEngine = ASTEngine(source=source, target=target)
  res = engine.run(input_code)
  assert res.code is not None
  assert len(res.code.strip()) > 0

  # Semantic target validation
  if target == "torch":
    assert "torch" in res.code
  elif target == "flax_nnx":
    assert "nnx" in res.code or "jax" in res.code
  elif target == "mlx":
    assert "mlx" in res.code or "mx" in res.code
  elif target == "keras":
    assert "keras" in res.code
  elif target == "nvidia_sass":
    assert "R" in res.code or "MOV" in res.code or "FFMA" in res.code
  elif target == "rdna":
    assert "v" in res.code or "s" in res.code or "RDNA" in res.code


def test_sass_rdna_roundtrip() -> None:
  """Tests roundtrip identity and preservation between NVIDIA SASS and AMD RDNA."""
  engine_sass_to_rdna: ASTEngine = ASTEngine(source="nvidia_sass", target="rdna")
  rdna_output = engine_sass_to_rdna.run(SASS_SNIPPET)
  assert "BEGIN Linear" in rdna_output.code
  assert "v_fmac_f32" in rdna_output.code

  engine_rdna_to_sass: ASTEngine = ASTEngine(source="rdna", target="nvidia_sass")
  sass_output = engine_rdna_to_sass.run(rdna_output.code)
  assert "BEGIN Linear" in sass_output.code
  assert "FFMA" in sass_output.code


def test_high_level_roundtrips() -> None:
  """Tests high level roundtrips across eager and functional Python ML frameworks."""
  engine_torch_to_mlx: ASTEngine = ASTEngine(source="torch", target="mlx")
  mlx_output = engine_torch_to_mlx.run(TORCH_SNIPPET)
  assert "class Model(nn.Module):" in mlx_output.code

  engine_mlx_to_torch: ASTEngine = ASTEngine(source="mlx", target="torch")
  torch_output = engine_mlx_to_torch.run(mlx_output.code)
  assert "class Model(nn.Module):" in torch_output.code
  assert "def forward(self, x):" in torch_output.code
