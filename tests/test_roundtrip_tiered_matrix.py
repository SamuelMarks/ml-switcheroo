"""Exhaustive Tiered Adapter Examples Testing Matrix Suite.

Systematically verifies direct and bidirectional transpilation for all tiered
example snippets defined across all registered framework adapters (Phase 5).
"""

from __future__ import annotations

import ast
from typing import List, Tuple

import pytest

from ml_switcheroo import convert
from ml_switcheroo.frameworks import get_adapter

pytestmark = pytest.mark.slow

TORCH_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "jax"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "keras"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "mlx"),
  ("tier1_math", "numpy"),
  ("tier1_math", "ir"),
  ("tier1_math", "mlir"),
  ("tier1_math", "stablehlo"),
  ("tier1_math", "nvidia_sass"),
  ("tier1_math", "rdna"),
  # tier2_neural_simple
  ("tier2_neural_simple", "jax"),
  ("tier2_neural_simple", "flax_nnx"),
  ("tier2_neural_simple", "keras"),
  ("tier2_neural_simple", "tensorflow"),
  ("tier2_neural_simple", "mlx"),
  ("tier2_neural_simple", "paxml"),
  ("tier2_neural_simple", "ir"),
  ("tier2_neural_simple", "html"),
  ("tier2_neural_simple", "latex_dsl"),
  ("tier2_neural_simple", "tikz"),
  # tier2_neural_cnn
  ("tier2_neural_cnn", "jax"),
  ("tier2_neural_cnn", "flax_nnx"),
  ("tier2_neural_cnn", "keras"),
  ("tier2_neural_cnn", "tensorflow"),
  ("tier2_neural_cnn", "mlx"),
  ("tier2_neural_cnn", "paxml"),
  ("tier2_neural_cnn", "ir"),
  ("tier2_neural_cnn", "nvidia_sass"),
  ("tier2_neural_cnn", "rdna"),
  # tier3_extras_dataloader
  ("tier3_extras_dataloader", "jax"),
  ("tier3_extras_dataloader", "flax_nnx"),
  ("tier3_extras_dataloader", "keras"),
  ("tier3_extras_dataloader", "tensorflow"),
  ("tier3_extras_dataloader", "mlx"),
  # tier4_qwen3
  ("tier4_qwen3", "jax"),
  ("tier4_qwen3", "flax_nnx"),
  ("tier4_qwen3", "keras"),
  ("tier4_qwen3", "tensorflow"),
  ("tier4_qwen3", "mlx"),
  ("tier4_qwen3", "ir"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", TORCH_TIER_MATRIX)
def test_torch_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests PyTorch adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the PyTorch tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("torch")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in torch adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="torch", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"jax", "flax_nnx", "keras", "tensorflow", "mlx", "numpy", "paxml"}:
    ast.parse(converted)


KERAS_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "mlx"),
  ("tier1_math", "numpy"),
  ("tier1_math", "ir"),
  # tier2_neural_sequential
  ("tier2_neural_sequential", "torch"),
  ("tier2_neural_sequential", "jax"),
  ("tier2_neural_sequential", "flax_nnx"),
  ("tier2_neural_sequential", "tensorflow"),
  ("tier2_neural_sequential", "mlx"),
  ("tier2_neural_sequential", "ir"),
  # tier3_extras_rng
  ("tier3_extras_rng", "torch"),
  ("tier3_extras_rng", "jax"),
  ("tier3_extras_rng", "flax_nnx"),
  ("tier3_extras_rng", "tensorflow"),
  ("tier3_extras_rng", "mlx"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", KERAS_TIER_MATRIX)
def test_keras_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests Keras adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the Keras tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("keras")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in keras adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="keras", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "flax_nnx", "tensorflow", "mlx", "numpy"}:
    ast.parse(converted)


TENSORFLOW_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "keras"),
  ("tier1_math", "mlx"),
  ("tier1_math", "numpy"),
  ("tier1_math", "ir"),
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "flax_nnx"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "flax_nnx"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "mlx"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", TENSORFLOW_TIER_MATRIX)
def test_tensorflow_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests TensorFlow adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the TensorFlow tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("tensorflow")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in tensorflow adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="tensorflow", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "flax_nnx", "keras", "mlx", "numpy"}:
    ast.parse(converted)


JAX_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "keras"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "mlx"),
  ("tier1_math", "numpy"),
  ("tier1_math", "ir"),
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "flax_nnx"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "tensorflow"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "flax_nnx"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "tensorflow"),
  ("tier3_extras", "mlx"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", JAX_TIER_MATRIX)
def test_jax_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests JAX adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the JAX tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("jax")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in jax adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="jax", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "flax_nnx", "keras", "tensorflow", "mlx", "numpy"}:
    ast.parse(converted)


FLAX_NNX_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "tensorflow"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "tensorflow"),
  ("tier3_extras", "mlx"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", FLAX_NNX_TIER_MATRIX)
def test_flax_nnx_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests Flax NNX adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the Flax NNX tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("flax_nnx")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in flax_nnx adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="flax_nnx", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "keras", "tensorflow", "mlx"}:
    ast.parse(converted)


MLX_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "keras"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "numpy"),
  ("tier1_math", "ir"),
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "flax_nnx"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "tensorflow"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "flax_nnx"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "tensorflow"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", MLX_TIER_MATRIX)
def test_mlx_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests MLX adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the MLX tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("mlx")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in mlx adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="mlx", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "flax_nnx", "keras", "tensorflow", "numpy"}:
    ast.parse(converted)


NUMPY_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "keras"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "mlx"),
  ("tier1_math", "ir"),
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "flax_nnx"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "tensorflow"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "tensorflow"),
  ("tier3_extras", "mlx"),
]


@pytest.mark.parametrize("tier_name,target_fw", NUMPY_TIER_MATRIX)
def test_numpy_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests NumPy adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the NumPy tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("numpy")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in numpy adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="numpy", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "flax_nnx", "keras", "tensorflow", "mlx"}:
    ast.parse(converted)


PAXML_TIER_MATRIX: List[Tuple[str, str]] = [
  # tier1_math
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "flax_nnx"),
  ("tier1_math", "keras"),
  ("tier1_math", "tensorflow"),
  ("tier1_math", "mlx"),
  ("tier1_math", "ir"),
  # tier2_neural
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "flax_nnx"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "tensorflow"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "ir"),
  # tier3_extras
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "flax_nnx"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "tensorflow"),
  ("tier3_extras", "mlx"),
  # tier4_qwen3-vl
  ("tier4_qwen3-vl", "torch"),
  ("tier4_qwen3-vl", "jax"),
  ("tier4_qwen3-vl", "flax_nnx"),
  ("tier4_qwen3-vl", "keras"),
  ("tier4_qwen3-vl", "tensorflow"),
  ("tier4_qwen3-vl", "mlx"),
  ("tier4_qwen3-vl", "ir"),
]


@pytest.mark.parametrize("tier_name,target_fw", PAXML_TIER_MATRIX)
def test_paxml_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests Paxml adapter tiered examples conversion across target frameworks.

  Args:
      tier_name (str): Identifier of the Paxml tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("paxml")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in paxml adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="paxml", target=target_fw)
  assert converted is not None
  assert len(converted) > 0
  if target_fw in {"torch", "jax", "flax_nnx", "keras", "tensorflow", "mlx"}:
    ast.parse(converted)


IR_TIER_MATRIX: List[Tuple[str, str]] = [
  ("tier1_math", "torch"),
  ("tier1_math", "jax"),
  ("tier1_math", "keras"),
  ("tier1_math", "mlx"),
  ("tier1_math", "mlir"),
  ("tier1_math", "stablehlo"),
  ("tier2_neural", "torch"),
  ("tier2_neural", "jax"),
  ("tier2_neural", "keras"),
  ("tier2_neural", "mlx"),
  ("tier2_neural", "mlir"),
  ("tier2_neural", "stablehlo"),
  ("tier3_extras", "torch"),
  ("tier3_extras", "jax"),
  ("tier3_extras", "keras"),
  ("tier3_extras", "mlx"),
  ("tier3_extras", "mlir"),
  ("tier3_extras", "stablehlo"),
]


@pytest.mark.parametrize("tier_name,target_fw", IR_TIER_MATRIX)
def test_ir_tiered_examples_conversion(tier_name: str, target_fw: str) -> None:
  """Tests IR adapter tiered examples conversion across target frameworks and dialects.

  Args:
      tier_name (str): Identifier of the IR tiered example.
      target_fw (str): Destination framework or dialect.
  """
  adapter = get_adapter("ir")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in ir adapter"
  snippet = tiered[tier_name]

  converted = convert(snippet, source="ir", target=target_fw)
  assert converted is not None
  assert len(converted) > 0


@pytest.mark.parametrize("tier_name", ["tier1_math", "tier2_neural", "tier3_extras"])
def test_mlir_tiered_examples_syntax(tier_name: str) -> None:
  """Tests MLIR adapter tiered examples syntax validity.

  Args:
      tier_name (str): Identifier of the MLIR tiered example.
  """
  adapter = get_adapter("mlir")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in mlir adapter"
  code = tiered[tier_name]
  assert len(code) > 0
  if tier_name != "tier3_extras":
    assert "sw.module" in code or "func.func" in code or "builtin.module" in code


@pytest.mark.parametrize("tier_name", ["tier1_math", "tier2_neural", "tier3_extras"])
def test_stablehlo_tiered_examples_syntax(tier_name: str) -> None:
  """Tests StableHLO adapter tiered examples syntax validity.

  Args:
      tier_name (str): Identifier of the StableHLO tiered example.
  """
  adapter = get_adapter("stablehlo")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in stablehlo adapter"
  code = tiered[tier_name]
  assert len(code) > 0
  if tier_name != "tier3_extras":
    assert "stablehlo" in code or "shlo" in code


@pytest.mark.parametrize(
  "tier_name",
  ["tier1_math", "tier2_neural_simple", "tier2_neural_cnn", "tier4_qwen3", "tier4_qwen3-vl"],
)
def test_nvidia_sass_tiered_examples_emission(tier_name: str) -> None:
  """Tests NVIDIA SASS adapter tiered examples assembly emission.

  Args:
      tier_name (str): Identifier of the NVIDIA SASS tiered example.
  """
  adapter = get_adapter("nvidia_sass")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in nvidia_sass adapter"
  code = tiered[tier_name]
  assert len(code) > 0


@pytest.mark.parametrize(
  "tier_name",
  [
    "tier1_math",
    "tier2_neural_simple",
    "tier2_neural_cnn",
    "tier3_extras",
    "tier4_qwen3",
    "tier4_qwen3-vl",
  ],
)
def test_rdna_tiered_examples_emission(tier_name: str) -> None:
  """Tests AMD RDNA adapter tiered examples assembly emission.

  Args:
      tier_name (str): Identifier of the AMD RDNA tiered example.
  """
  adapter = get_adapter("rdna")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered, f"Missing {tier_name} in rdna adapter"
  code = tiered[tier_name]
  assert len(code) > 0


def test_html_dsl_tiered_examples_validation() -> None:
  """Tests HTML DSL adapter tiered example DOM/SVG structure validation."""
  adapter = get_adapter("html")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert "tier2_neural" in tiered
  code = tiered["tier2_neural"]
  assert "<" in code and ">" in code


@pytest.mark.parametrize("tier_name", ["tier1_math", "tier2_neural", "tier3_extras"])
def test_latex_dsl_tiered_examples_validation(tier_name: str) -> None:
  """Tests LaTeX DSL adapter tiered examples validation.

  Args:
      tier_name (str): Identifier of the LaTeX DSL tiered example.
  """
  adapter = get_adapter("latex_dsl")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered
  code = tiered[tier_name]
  assert len(code) > 0


@pytest.mark.parametrize("tier_name", ["tier1_math", "tier2_neural", "tier3_extras"])
def test_tikz_tiered_examples_validation(tier_name: str) -> None:
  """Tests TikZ adapter tiered examples validation.

  Args:
      tier_name (str): Identifier of the TikZ tiered example.
  """
  adapter = get_adapter("tikz")
  assert adapter is not None
  tiered = adapter.get_tiered_examples()
  assert tier_name in tiered
  code = tiered[tier_name]
  assert len(code) > 0
