"""Comprehensive Exhaustive Roundtrip Test Suite.

Validates that high-level frameworks (PyTorch, JAX, Flax NNX, Apple MLX, Keras 3,
TensorFlow, NumPy, PaxML) and low-level targets (NVIDIA SASS, AMD RDNA, MLIR,
ML-Switcheroo IR) roundtrip idempotently without spurious argument accumulation.
"""

import ast

import pytest

from ml_switcheroo import convert

HIGH_LEVEL_FRAMEWORKS: list[str] = [
  "torch",
  "jax",
  "flax_nnx",
  "keras",
  "mlx",
  "tensorflow",
  "numpy",
  "paxml",
]

HARDWARE_AND_IR_TARGETS: list[str] = [
  "nvidia_sass",
  "rdna",
  "ml_switcheroo_ir",
  "latex_dsl",
  "html",
]

_TARGET_IDIOM_MARKERS: dict[str, tuple[str, ...]] = {
  "torch": ("torch", "nn."),
  "jax": ("jnp.", "jax", "flax", "nn."),
  "flax_nnx": ("nnx.", "jnp.", "jax"),
  "keras": ("keras", "ops.", "layers."),
  "mlx": ("mx.", "mlx", "nn."),
  "tensorflow": ("tf.", "keras"),
  "numpy": ("np.", "numpy"),
  "paxml": ("praxis", "pl.", "jnp.", "jax"),
}

_SOURCE_ROOT_IMPORT: dict[str, str] = {
  "torch": "import torch\n",
  "mlx": "import mlx\n",
  "tensorflow": "import tensorflow\n",
  "paxml": "import praxis\n",
}

# Canonical representative model snippets
CANONICAL_SNIPPETS: dict[str, str] = {
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


def _extract_ast_signature(code: str) -> tuple[set[str], set[str]]:
  """Extract class and top-level function names from Python code.

  Args:
      code: Python source code string.

  Returns:
      Tuple of (class_names, function_names).
  """
  tree = ast.parse(code)
  classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
  funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
  return classes, funcs


def _get_all_framework_pairs() -> list[tuple[str, str]]:
  """Generate all non-identical ordered pairs of high-level frameworks.

  Returns:
      List of (source_framework, target_framework) string tuples.
  """
  pairs: list[tuple[str, str]] = []
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
  orig_classes, orig_funcs = _extract_ast_signature(initial_torch_code)

  # 1. Establish source representation if source is not PyTorch
  if src == "torch":
    source_code = initial_torch_code
  else:
    source_code = convert(initial_torch_code, source="torch", target=src)
    src_classes, src_funcs = _extract_ast_signature(source_code)
    assert orig_classes == src_classes
    assert orig_funcs == src_funcs

  # 2. Forward Transpilation (src -> tgt)
  fwd_code = convert(source_code, source=src, target=tgt)
  fwd_classes, fwd_funcs = _extract_ast_signature(fwd_code)
  assert orig_classes == fwd_classes, f"Classes changed in {src}->{tgt}: {orig_classes} vs {fwd_classes}"
  assert orig_funcs == fwd_funcs, f"Functions changed in {src}->{tgt}: {orig_funcs} vs {fwd_funcs}"
  tgt_markers = _TARGET_IDIOM_MARKERS.get(tgt, ())
  if tgt_markers and (snippet_key == "math_ops" or "numpy" not in (src, tgt)):
    assert any(m in fwd_code for m in tgt_markers), f"Target {tgt} markers {tgt_markers} missing from:\n{fwd_code}"
  forbidden_src_import = _SOURCE_ROOT_IMPORT.get(src)
  if forbidden_src_import and tgt not in ("jax", "flax_nnx", "paxml"):
    assert forbidden_src_import not in fwd_code, f"Source import '{forbidden_src_import}' leaked into {tgt}:\n{fwd_code}"

  # 3. Backward Transpilation (tgt -> src)
  bwd_code = convert(fwd_code, source=tgt, target=src)
  bwd_classes, bwd_funcs = _extract_ast_signature(bwd_code)
  assert orig_classes == bwd_classes, f"Classes changed in roundtrip {src}->{tgt}->{src}"
  assert orig_funcs == bwd_funcs, f"Functions changed in roundtrip {src}->{tgt}->{src}"
  src_markers = _TARGET_IDIOM_MARKERS.get(src, ())
  if src_markers and (snippet_key == "math_ops" or "numpy" not in (src, tgt)):
    assert any(m in bwd_code for m in src_markers), f"Source {src} markers {src_markers} missing from:\n{bwd_code}"

  # 4. Idempotency & Structural Equivalence Check:
  # Translating backward code once more produces structurally stable code
  second_fwd_code = convert(bwd_code, source=src, target=tgt)
  sec_classes, sec_funcs = _extract_ast_signature(second_fwd_code)
  assert sec_classes == fwd_classes
  assert sec_funcs == fwd_funcs


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
  bwd_tree = ast.parse(bwd)
  assert len(bwd_tree.body) > 0
  assert "torch" in bwd or "def " in bwd or "class " in bwd


@pytest.mark.parametrize("spk", ["torch", "jax", "mlx", "keras"])
def test_neural_model_ir_roundtrip(spk: str) -> None:
  """Verifies that neural model graphs roundtrip through ML-Switcheroo IR.

  Args:
      spk: Framework spoke identifier.
  """
  code = CANONICAL_SNIPPETS["mlp"]
  src_code = code if spk == "torch" else convert(code, source="torch", target=spk)
  fwd = convert(src_code, source=spk, target="ml_switcheroo_ir")
  assert "LogicalGraph" in fwd and "SimpleMLP" in fwd

  bwd = convert(fwd, source="ml_switcheroo_ir", target=spk)
  bwd_classes, _ = _extract_ast_signature(bwd)
  assert "SimpleMLP" in bwd_classes
  markers = _TARGET_IDIOM_MARKERS.get(spk, ())
  if markers:
    assert any(m in bwd for m in markers)


def test_cross_isa_assembly_roundtrip() -> None:
  """Verifies that assembly representations transpile bidirectionally between NVIDIA SASS and AMD RDNA."""
  code = CANONICAL_SNIPPETS["math_ops"]
  sass = convert(code, source="torch", target="nvidia_sass")
  assert "fadd" in sass.lower() or "fabs" in sass.lower(), "Expected SASS instructions in emitted assembly."

  rdna = convert(sass, source="nvidia_sass", target="rdna")
  assert "v_add" in rdna.lower() or "bb" in rdna.lower() or "mean" in rdna.lower(), (
    "Expected RDNA instructions in transpiled assembly."
  )

  sass_back = convert(rdna, source="rdna", target="nvidia_sass")
  assert "fadd" in sass_back.lower() or "fabs" in sass_back.lower() or "//" in sass_back


def test_neural_hardware_isa_decompilation() -> None:
  """Verifies that high-level neural modules lower to SASS and RDNA and decompile back to valid AST."""
  code = CANONICAL_SNIPPETS["mlp"]

  # 1. SASS
  sass = convert(code, source="torch", target="nvidia_sass")
  assert "BEGIN Linear" in sass
  torch_from_sass = convert(sass, source="nvidia_sass", target="torch")
  sass_classes, _ = _extract_ast_signature(torch_from_sass)
  assert len(sass_classes) >= 1
  assert "nn.Linear" in torch_from_sass

  # 2. RDNA
  rdna = convert(code, source="torch", target="rdna")
  assert "BEGIN Linear" in rdna
  torch_from_rdna = convert(rdna, source="rdna", target="torch")
  rdna_classes, _ = _extract_ast_signature(torch_from_rdna)
  assert len(rdna_classes) >= 1
  assert "nn.Linear" in torch_from_rdna


def test_flatten_no_spurious_default_arguments() -> None:
  """Verifies that torch.flatten(x, 1) does not accumulate end_dim=-1 across roundtrip hops."""
  code = CANONICAL_SNIPPETS["conv_net"]
  fwd = convert(code, source="torch", target="flax_nnx")
  assert "end_dim=-1" not in fwd, f"Forward transpilation injected end_dim into JAX/NNX: {fwd}"

  bwd = convert(fwd, source="flax_nnx", target="torch")
  assert "end_dim=-1" not in bwd, f"Backward transpilation injected end_dim into PyTorch: {bwd}"
