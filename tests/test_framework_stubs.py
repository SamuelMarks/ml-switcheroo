"""Tests for generated framework mapping stubs."""

import pytest
from pathlib import Path
from ml_switcheroo.semantics.schema import validate_yaml_schema

STUBS_DIR = Path("src/ml_switcheroo/semantics/odl")


@pytest.mark.parametrize(
  "stub_file, expected_frameworks",
  [
    ("__framework_jax.yaml", ["jax", "flax"]),
    ("__framework_torch.yaml", ["torch", "torch_nn"]),
    ("__framework_mlx.yaml", ["mlx", "mlx_nn"]),
    ("__framework_keras.yaml", ["keras", "keras_layers"]),
  ],
)
def test_framework_stubs(stub_file, expected_frameworks):
  """Test that each framework stub is valid and contains expected keys."""
  file_path = STUBS_DIR / stub_file
  assert file_path.exists(), f"{stub_file} missing"

  content = file_path.read_text(encoding="utf-8")
  result = validate_yaml_schema(content)

  assert result.frameworks is not None
  for fw in expected_frameworks:
    assert fw in result.frameworks
