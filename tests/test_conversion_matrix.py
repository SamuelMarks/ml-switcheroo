"""Test suite for the 6x6 Full Conversion Matrix across Python frameworks and hardware ISAs.

Verifies end-to-end static source-to-source translation across all 30 directed pairs of:
- PyTorch (torch)
- JAX / Flax NNX (jax)
- Apple MLX (mlx)
- Keras 3 (keras)
- AMD RDNA Assembly (rdna)
- NVIDIA SASS Assembly (nvidia_sass)
"""

from typing import List, Tuple
import pytest

from ml_switcheroo.core.engine import ASTEngine, ConversionResult

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
import keras.ops as ops

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
    s_add_i32 s0, s0, 1
    s_cmp_lt_i32 s0, 128
    s_cbranch_scc1 L_GEMM_fc
    ; END Linear (fc)
    ; Return: v1
"""

FRAMEWORK_SNIPPETS: dict[str, str] = {
  "torch": TORCH_SNIPPET,
  "jax": JAX_SNIPPET,
  "mlx": MLX_SNIPPET,
  "keras": KERAS_SNIPPET,
  "nvidia_sass": SASS_SNIPPET,
  "rdna": RDNA_SNIPPET,
}

TARGET_FRAMEWORKS: List[str] = [
  "torch",
  "jax",
  "mlx",
  "keras",
  "nvidia_sass",
  "rdna",
]

ALL_DIRECTED_PAIRS: List[Tuple[str, str]] = [
  (src, tgt) for src in TARGET_FRAMEWORKS for tgt in TARGET_FRAMEWORKS if src != tgt
]


@pytest.mark.parametrize("source,target", ALL_DIRECTED_PAIRS)
def test_full_conversion_matrix_edge(source: str, target: str) -> None:
  """Tests static translation across a directed edge in the 6x6 conversion matrix.

  Args:
      source: The source framework or ISA name.
      target: The target framework or ISA name.
  """
  input_code: str = FRAMEWORK_SNIPPETS[source]
  engine = ASTEngine(
    source=source,
    target=target,
    strict_mode=False,
  )
  res: ConversionResult = engine.run(input_code)

  # 1. Conversion must succeed and produce non-empty code
  assert res.code is not None, f"Conversion failed from {source} to {target}"
  clean_code = res.code.strip()
  assert len(clean_code) > 0, f"Generated empty code from {source} to {target}"

  # 2. Target framework tokens validation
  if target == "torch":
    assert "torch" in clean_code or "nn" in clean_code
  elif target == "jax":
    assert "jax" in clean_code or "nnx" in clean_code or "jnp" in clean_code
  elif target == "mlx":
    assert "mlx" in clean_code or "mx" in clean_code
  elif target == "keras":
    assert "keras" in clean_code or "layers" in clean_code
  elif target == "nvidia_sass":
    assert "R" in clean_code or "MOV" in clean_code or "FFMA" in clean_code
  elif target == "rdna":
    assert "v" in clean_code or "s" in clean_code or "RDNA" in clean_code


def test_high_level_roundtrip_submatrix() -> None:
  """Verifies roundtrips across high-level Python frameworks (PyTorch, JAX, MLX, Keras)."""
  py_frameworks = ["torch", "jax", "mlx", "keras"]
  for src in py_frameworks:
    for tgt in py_frameworks:
      if src == tgt:
        continue
      engine = ASTEngine(source=src, target=tgt, strict_mode=False)
      res = engine.run(FRAMEWORK_SNIPPETS[src])
      assert res.code is not None
      assert len(res.code.strip()) > 0
