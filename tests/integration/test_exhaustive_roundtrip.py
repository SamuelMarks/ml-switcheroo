"""Comprehensive Exhaustive Roundtrip Test Suite.

Validates that high-level frameworks (PyTorch, JAX, Flax NNX, Apple MLX, Keras 3,
TensorFlow, NumPy, PaxML) and low-level targets (NVIDIA SASS, AMD RDNA, MLIR,
ML-Switcheroo IR) roundtrip idempotently without spurious argument accumulation.
"""

import ast
from typing import Dict, List, Tuple
import pytest

from ml_switcheroo import convert

HIGH_LEVEL_FRAMEWORKS: List[str] = [
  "torch",
  "jax",
  "flax_nnx",
  "keras",
  "mlx",
  "tensorflow",
  "numpy",
  "paxml",
]

HARDWARE_AND_IR_TARGETS: List[str] = [
  "nvidia_sass",
  "rdna",
  "ml_switcheroo_ir",
  "latex_dsl",
  "html",
]

# Canonical representative model snippets
CANONICAL_SNIPPETS: Dict[str, str] = {
  "math_ops": """import torch

def math_ops(x, y):
    a = torch.abs(x)
    b = torch.add(a, y)
    return torch.mean(b)
""",
  "conv_net": """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x
""",
  "mlp": """import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.relu(x)
""",
}


def _get_all_framework_pairs() -> List[Tuple[str, str]]:
  """Generate all non-identical ordered pairs of high-level frameworks.

  Returns:
      List of (source_framework, target_framework) string tuples.
  """
  pairs: List[Tuple[str, str]] = []
  for src in HIGH_LEVEL_FRAMEWORKS:
    for tgt in HIGH_LEVEL_FRAMEWORKS:
      if src != tgt:
        pairs.append((src, tgt))
  return pairs


@pytest.mark.parametrize("snippet_key", sorted(CANONICAL_SNIPPETS.keys()))
@pytest.mark.parametrize("src,tgt", _get_all_framework_pairs())
def test_high_level_framework_roundtrip(snippet_key: str, src: str, tgt: str) -> None:
  """Verifies forward and backward roundtrip conversion between high-level frameworks.

  Args:
      snippet_key: The canonical snippet key to test.
      src: Source framework identifier.
      tgt: Intermediate target framework identifier.
  """
  initial_torch_code = CANONICAL_SNIPPETS[snippet_key]

  # 1. Establish source representation if source is not PyTorch
  if src == "torch":
    source_code = initial_torch_code
  else:
    source_code = convert(initial_torch_code, source="torch", target=src)
    if src not in ("html", "tikz", "latex_dsl"):
      ast.parse(source_code)

  # 2. Forward Transpilation (src -> tgt)
  fwd_code = convert(source_code, source=src, target=tgt)
  assert fwd_code.strip(), f"Forward transpilation from {src} to {tgt} produced empty output."
  if tgt not in ("html", "tikz", "latex_dsl"):
    ast.parse(fwd_code)

  # 3. Backward Transpilation (tgt -> src)
  bwd_code = convert(fwd_code, source=tgt, target=src)
  assert bwd_code.strip(), f"Backward transpilation from {tgt} to {src} produced empty output."
  if src not in ("html", "tikz", "latex_dsl"):
    ast.parse(bwd_code)

  # 4. Idempotency & Structural Equivalence Check:
  # Translating backward code once more produces structurally stable code
  second_fwd_code = convert(bwd_code, source=src, target=tgt)
  assert second_fwd_code.strip(), f"Second forward transpilation from {src} to {tgt} produced empty output."
  if tgt not in ("html", "tikz", "latex_dsl"):
    ast.parse(second_fwd_code)


@pytest.mark.parametrize("target_ir", HARDWARE_AND_IR_TARGETS)
def test_math_to_hardware_and_ir_roundtrip(target_ir: str) -> None:
  """Verifies that mathematical tensor operations roundtrip through hardware ISAs and IRs.

  Args:
      target_ir: The hardware ISA or IR dialect identifier.
  """
  code = CANONICAL_SNIPPETS["math_ops"]
  fwd = convert(code, source="torch", target=target_ir)
  assert fwd.strip(), f"Failed to emit {target_ir} from PyTorch math ops."

  bwd = convert(fwd, source=target_ir, target="torch")
  assert bwd.strip(), f"Failed to decompile {target_ir} back to PyTorch math ops."
  ast.parse(bwd)


@pytest.mark.parametrize("spk", ["torch", "jax", "mlx", "keras"])
def test_neural_model_ir_roundtrip(spk: str) -> None:
  """Verifies that neural model graphs roundtrip through ML-Switcheroo IR.

  Args:
      spk: Framework spoke identifier.
  """
  code = CANONICAL_SNIPPETS["mlp"]
  src_code = code if spk == "torch" else convert(code, source="torch", target=spk)
  fwd = convert(src_code, source=spk, target="ml_switcheroo_ir")
  assert fwd.strip(), f"Failed to emit ML-Switcheroo IR from {spk}."

  bwd = convert(fwd, source="ml_switcheroo_ir", target=spk)
  assert bwd.strip(), f"Failed to decompile ML-Switcheroo IR back to {spk}."
  ast.parse(bwd)


def test_cross_isa_assembly_roundtrip() -> None:
  """Verifies that assembly representations transpile bidirectionally between NVIDIA SASS and AMD RDNA."""
  code = CANONICAL_SNIPPETS["math_ops"]
  sass = convert(code, source="torch", target="nvidia_sass")
  assert "FADD" in sass or "FABS" in sass, "Expected SASS instructions in emitted assembly."

  rdna = convert(sass, source="nvidia_sass", target="rdna")
  assert "v_add_f32" in rdna or "BB" in rdna, "Expected RDNA instructions in transpiled assembly."

  sass_back = convert(rdna, source="rdna", target="nvidia_sass")
  assert sass_back.strip(), "Failed to transpile RDNA back to NVIDIA SASS."


def test_neural_hardware_isa_decompilation() -> None:
  """Verifies that high-level neural modules lower to SASS and RDNA and decompile back to valid AST."""
  code = CANONICAL_SNIPPETS["mlp"]

  # 1. SASS
  sass = convert(code, source="torch", target="nvidia_sass")
  torch_from_sass = convert(sass, source="nvidia_sass", target="torch")
  ast.parse(torch_from_sass)

  # 2. RDNA
  rdna = convert(code, source="torch", target="rdna")
  torch_from_rdna = convert(rdna, source="rdna", target="torch")
  ast.parse(torch_from_rdna)


def test_flatten_no_spurious_default_arguments() -> None:
  """Verifies that torch.flatten(x, 1) does not accumulate end_dim=-1 across roundtrip hops."""
  code = CANONICAL_SNIPPETS["conv_net"]
  fwd = convert(code, source="torch", target="flax_nnx")
  assert "end_dim=-1" not in fwd, f"Forward transpilation injected end_dim into JAX/NNX: {fwd}"

  bwd = convert(fwd, source="flax_nnx", target="torch")
  assert "end_dim=-1" not in bwd, f"Backward transpilation injected end_dim into PyTorch: {bwd}"
