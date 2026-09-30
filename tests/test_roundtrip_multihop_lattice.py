"""Exhaustive Multi-Hop and Lattice Roundtrip Testing Suite.

Systematically verifies cyclic triangular lattices and diamond multi-hop conversions
across frameworks, intermediate representations, and hardware targets (Phase 6).
"""

from __future__ import annotations

import ast

from ml_switcheroo import convert
from ml_switcheroo.frameworks import get_adapter


def test_triangle_1_torch_jax_keras_torch_math() -> None:
  """Tests Triangular lattice 1 with math operations: torch -> jax -> keras -> torch."""
  torch_adapter = get_adapter("torch")
  assert torch_adapter is not None
  code = torch_adapter.get_tiered_examples()["tier1_math"]

  # Step 1: torch -> jax
  step1 = convert(code, source="torch", target="jax")
  ast.parse(step1)
  assert "torch" not in step1

  # Step 2: jax -> keras
  step2 = convert(step1, source="jax", target="keras")
  ast.parse(step2)

  # Step 3: keras -> torch
  step3 = convert(step2, source="keras", target="torch")
  parsed = ast.parse(step3)
  assert parsed is not None
  assert "torch" in step3


def test_triangle_1_torch_jax_keras_torch_neural() -> None:
  """Tests Triangular lattice 1 with neural network: torch -> jax -> keras -> torch."""
  torch_adapter = get_adapter("torch")
  assert torch_adapter is not None
  code = torch_adapter.get_tiered_examples()["tier2_neural_simple"]

  # Step 1: torch -> jax
  step1 = convert(code, source="torch", target="jax")
  ast.parse(step1)

  # Step 2: jax -> keras
  step2 = convert(step1, source="jax", target="keras")
  ast.parse(step2)

  # Step 3: keras -> torch
  step3 = convert(step2, source="keras", target="torch")
  parsed = ast.parse(step3)
  assert parsed is not None
  assert "nn.Module" in step3


def test_triangle_2_torch_mlx_tensorflow_torch() -> None:
  """Tests Triangular lattice 2: torch -> mlx -> tensorflow -> torch."""
  torch_adapter = get_adapter("torch")
  assert torch_adapter is not None
  code = torch_adapter.get_tiered_examples()["tier1_math"]

  step1 = convert(code, source="torch", target="mlx")
  ast.parse(step1)

  step2 = convert(step1, source="mlx", target="tensorflow")
  ast.parse(step2)

  step3 = convert(step2, source="tensorflow", target="torch")
  parsed = ast.parse(step3)
  assert parsed is not None
  assert "torch" in step3


def test_triangle_3_keras_jax_flax_nnx_keras() -> None:
  """Tests Triangular lattice 3: keras -> jax -> flax_nnx -> keras."""
  keras_adapter = get_adapter("keras")
  assert keras_adapter is not None
  code = keras_adapter.get_tiered_examples()["tier2_neural_sequential"]

  step1 = convert(code, source="keras", target="jax")
  ast.parse(step1)

  step2 = convert(step1, source="jax", target="flax_nnx")
  ast.parse(step2)

  step3 = convert(step2, source="flax_nnx", target="keras")
  parsed = ast.parse(step3)
  assert parsed is not None
  assert "keras" in step3


def test_triangle_4_numpy_torch_jax_numpy() -> None:
  """Tests Triangular lattice 4: numpy -> torch -> jax -> numpy."""
  np_adapter = get_adapter("numpy")
  assert np_adapter is not None
  code = np_adapter.get_tiered_examples()["tier1_math"]

  step1 = convert(code, source="numpy", target="torch")
  ast.parse(step1)

  step2 = convert(step1, source="torch", target="jax")
  ast.parse(step2)

  step3 = convert(step2, source="jax", target="numpy")
  parsed = ast.parse(step3)
  assert parsed is not None
  assert "numpy" in step3 or "np" in step3


def test_diamond_1_torch_jax_vs_keras_to_ir() -> None:
  """Tests Diamond lattice 1: torch -> jax -> ir vs torch -> keras -> ir."""
  torch_adapter = get_adapter("torch")
  assert torch_adapter is not None
  code = torch_adapter.get_tiered_examples()["tier1_math"]

  # Path A: torch -> jax -> ir
  jax_code = convert(code, source="torch", target="jax")
  ir_a = convert(jax_code, source="jax", target="ir")

  # Path B: torch -> keras -> ir
  keras_code = convert(code, source="torch", target="keras")
  ir_b = convert(keras_code, source="keras", target="ir")

  assert ir_a is not None and len(ir_a) > 0
  assert ir_b is not None and len(ir_b) > 0


def test_diamond_2_flax_nnx_torch_vs_ir_to_nvidia_sass() -> None:
  """Tests Diamond lattice 2: flax_nnx -> torch -> nvidia_sass vs flax_nnx -> ir -> nvidia_sass."""
  flax_adapter = get_adapter("flax_nnx")
  assert flax_adapter is not None
  code = flax_adapter.get_tiered_examples()["tier2_neural"]

  # Path A: flax_nnx -> torch -> nvidia_sass
  torch_code = convert(code, source="flax_nnx", target="torch")
  sass_a = convert(torch_code, source="torch", target="nvidia_sass")

  # Path B: flax_nnx -> ir -> nvidia_sass
  ir_code = convert(code, source="flax_nnx", target="ir")
  sass_b = convert(ir_code, source="ir", target="nvidia_sass")

  assert sass_a is not None and len(sass_a) > 0
  assert sass_b is not None and len(sass_b) > 0
