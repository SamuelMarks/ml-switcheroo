"""Exhaustive Roundtrip and Matrix Testing Suite.

Systematically validates bidirectional and multi-hop transpilation across all
canonical example files in `tests/examples/` and tiered framework examples.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from ml_switcheroo import convert
from ml_switcheroo.frameworks import available_frameworks, get_adapter

pytestmark = pytest.mark.slow

EXAMPLES_DIR: Path = Path(__file__).parent / "examples"

PYTHON_FRAMEWORKS: List[str] = [
  "torch",
  "jax",
  "flax_nnx",
  "keras",
  "tensorflow",
  "mlx",
  "numpy",
  "paxml",
]

IR_FRAMEWORKS: List[str] = [
  "ir",
  "ml_switcheroo_ir",
  "mlir",
  "stablehlo",
]

ASM_FRAMEWORKS: List[str] = [
  "nvidia_sass",
  "rdna",
]

DSL_FRAMEWORKS: List[str] = [
  "html",
  "latex_dsl",
  "tikz",
]


def load_canonical_example_files() -> List[Tuple[str, str, Path]]:
  """Discover all Python example files in the tests/examples directory.

  Returns:
      List[Tuple[str, str, Path]]: Tuples containing (example_name, source_framework, file_path).
  """
  if not EXAMPLES_DIR.is_dir():
    return []
  examples: List[Tuple[str, str, Path]] = []
  for p in sorted(EXAMPLES_DIR.glob("*.py")):
    if p.name == "__init__.py":
      continue
    parts = p.name.split(".")
    if len(parts) >= 3:
      source_fw = parts[-2]
      examples.append((p.name, source_fw, p))
  return examples


def load_all_adapter_tiered_examples() -> List[Tuple[str, str, str]]:
  """Discover all tiered examples provided by registered framework adapters.

  Returns:
      List[Tuple[str, str, str]]: Tuples containing (framework_name, tier_name, snippet_code).
  """
  results: List[Tuple[str, str, str]] = []
  for fw_name in sorted(available_frameworks()):
    try:
      adapter = get_adapter(fw_name)
      if adapter is not None and hasattr(adapter, "get_tiered_examples"):
        tiered: Dict[str, str] = adapter.get_tiered_examples()
        for tier_name, code in tiered.items():
          results.append((fw_name, tier_name, code))
    except Exception:
      pass
  return results


def test_examples_discovery() -> None:
  """Verifies that canonical example files are correctly discovered."""
  examples = load_canonical_example_files()
  assert len(examples) >= 10, f"Expected at least 10 canonical example files, found {len(examples)}"
  example_names = [e[0] for e in examples]
  assert "ex01_math_ops.torch.py" in example_names
  assert "ex01_math_ops.jax.py" in example_names
  assert "ex02_neural_net.torch.py" in example_names


def test_tiered_examples_discovery() -> None:
  """Verifies that framework adapters expose valid tiered examples."""
  tiered = load_all_adapter_tiered_examples()
  assert len(tiered) >= 20, f"Expected at least 20 tiered examples, found {len(tiered)}"
  frameworks = {t[0] for t in tiered}
  assert "torch" in frameworks
  assert "keras" in frameworks
  assert "jax" in frameworks


@pytest.mark.parametrize(
  "source_target",
  [
    ("torch", "jax"),
    ("jax", "torch"),
    ("torch", "flax_nnx"),
    ("flax_nnx", "torch"),
    ("torch", "keras"),
    ("keras", "torch"),
    ("torch", "mlx"),
    ("mlx", "torch"),
    ("jax", "keras"),
    ("keras", "jax"),
  ],
)
def test_math_ops_canonical_roundtrip(source_target: Tuple[str, str]) -> None:
  """Tests bidirectional conversion of math operations between core framework pairs.

  Args:
      source_target (Tuple[str, str]): Source framework and target framework pair.
  """
  source_fw, target_fw = source_target
  math_code = """def compute_loss(x, y):
    diff = x - y
    return diff
"""
  converted = convert(math_code, source=source_fw, target=target_fw)
  assert converted is not None
  parsed = ast.parse(converted)
  assert parsed is not None

  roundtripped = convert(converted, source=target_fw, target=source_fw)
  assert roundtripped is not None
  parsed_rt = ast.parse(roundtripped)
  assert parsed_rt is not None


@pytest.mark.parametrize(
  "triangle",
  [
    ("torch", "jax", "torch"),
    ("torch", "flax_nnx", "torch"),
    ("torch", "mlx", "torch"),
    ("jax", "torch", "jax"),
  ],
)
def test_triangular_math_roundtrip(triangle: Tuple[str, str, str]) -> None:
  """Tests cyclic roundtrip stability across frameworks.

  Args:
      triangle (Tuple[str, str, str]): Three-step cyclic conversion path.
  """
  f1, f2, f3 = triangle
  code = """import torch

def math_kernel(a, b):
    diff = torch.abs(a - b)
    return torch.mean(diff)
"""
  if f1 == "jax":
    code = """import jax.numpy as jnp

def math_kernel(a, b):
    diff = jnp.abs(a - b)
    return jnp.mean(diff)
"""
  step1 = convert(code, source=f1, target=f2)
  ast.parse(step1)
  step2 = convert(step1, source=f2, target=f3)
  parsed_final = ast.parse(step2)
  assert parsed_final is not None


def get_all_canonical_roundtrip_pairs() -> List[Tuple[str, str]]:
  """Generate all canonical example file to target framework pairs.

  Returns:
      List[Tuple[str, str]]: (example_filename, target_framework) pairs.
  """
  pairs: List[Tuple[str, str]] = []
  if not EXAMPLES_DIR.is_dir():
    return pairs
  for p in sorted(EXAMPLES_DIR.glob("*.py")):
    if p.name == "__init__.py":
      continue
    parts = p.name.split(".")
    if len(parts) >= 3:
      source_fw = parts[-2]
      for tgt in PYTHON_FRAMEWORKS:
        if tgt != source_fw:
          pairs.append((p.name, tgt))
  return pairs


def get_all_canonical_backend_pairs() -> List[Tuple[str, str]]:
  """Generate all canonical example file to backend target pairs.

  Returns:
      List[Tuple[str, str]]: (example_filename, backend_name) pairs.
  """
  pairs: List[Tuple[str, str]] = []
  if not EXAMPLES_DIR.is_dir():
    return pairs
  all_backends = ["ir", "mlir", "stablehlo", "nvidia_sass", "rdna", "html", "latex_dsl", "tikz"]
  for p in sorted(EXAMPLES_DIR.glob("*.py")):
    if p.name == "__init__.py":
      continue
    parts = p.name.split(".")
    if len(parts) >= 3:
      for b in all_backends:
        pairs.append((p.name, b))
  return pairs


CANONICAL_ROUNDTRIP_PAIRS: List[Tuple[str, str]] = get_all_canonical_roundtrip_pairs()
CANONICAL_BACKEND_PAIRS: List[Tuple[str, str]] = get_all_canonical_backend_pairs()


@pytest.mark.parametrize("example_name,target_fw", CANONICAL_ROUNDTRIP_PAIRS)
def test_canonical_file_example_roundtrip(example_name: str, target_fw: str) -> None:
  """Tests bidirectional conversion of canonical example files to target frameworks and back.

  Args:
      example_name (str): The filename of the canonical example.
      target_fw (str): The target framework identifier.
  """
  file_path = EXAMPLES_DIR / example_name
  if not file_path.is_file():
    pytest.skip(f"Example file {example_name} not found.")

  source_code = file_path.read_text(encoding="utf-8")
  source_fw = example_name.split(".")[-2]

  # 1. Forward conversion
  forward_code = convert(source_code, source=source_fw, target=target_fw)
  assert forward_code is not None
  forward_parsed = ast.parse(forward_code)
  assert forward_parsed is not None

  # 2. Reverse roundtrip conversion
  roundtripped_code = convert(forward_code, source=target_fw, target=source_fw)
  assert roundtripped_code is not None
  roundtripped_parsed = ast.parse(roundtripped_code)
  assert roundtripped_parsed is not None


@pytest.mark.parametrize("example_name,backend_target", CANONICAL_BACKEND_PAIRS)
def test_canonical_file_example_to_backend(example_name: str, backend_target: str) -> None:
  """Tests compilation of canonical example files to intermediate, assembly, and DSL targets.

  Args:
      example_name (str): The filename of the canonical example.
      backend_target (str): The backend framework identifier (ir, mlir, stablehlo, sass, rdna, html, latex, tikz).
  """
  file_path = EXAMPLES_DIR / example_name
  if not file_path.is_file():
    pytest.skip(f"Example file {example_name} not found.")

  source_code = file_path.read_text(encoding="utf-8")
  source_fw = example_name.split(".")[-2]

  compiled = convert(source_code, source=source_fw, target=backend_target)
  assert compiled is not None
  assert len(compiled) > 0


@pytest.mark.parametrize(
  "fw_name,tier_name,target_fw",
  [
    ("torch", "tier1_math", "jax"),
    ("torch", "tier1_math", "flax_nnx"),
    ("torch", "tier1_math", "mlx"),
    ("torch", "tier2_neural_simple", "flax_nnx"),
    ("keras", "tier1_math", "torch"),
    ("keras", "tier1_math", "jax"),
    ("jax", "tier1_math", "torch"),
    ("flax_nnx", "tier2_neural", "torch"),
    ("mlx", "tier1_math", "torch"),
    ("numpy", "tier1_math", "torch"),
  ],
)
def test_adapter_tiered_examples_roundtrip(fw_name: str, tier_name: str, target_fw: str) -> None:
  """Tests bidirectional roundtrip of framework adapter tiered examples.

  Args:
      fw_name (str): Source framework name.
      tier_name (str): Tier identifier name.
      target_fw (str): Target framework name.
  """
  adapter = get_adapter(fw_name)
  assert adapter is not None
  assert hasattr(adapter, "get_tiered_examples")
  tiered = adapter.get_tiered_examples()
  if tier_name not in tiered:
    pytest.skip(f"Tier {tier_name} not present in {fw_name}")

  snippet = tiered[tier_name]
  step1 = convert(snippet, source=fw_name, target=target_fw)
  ast.parse(step1)
  step2 = convert(step1, source=target_fw, target=fw_name)
  parsed_rt = ast.parse(step2)
  assert parsed_rt is not None
