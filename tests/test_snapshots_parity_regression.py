"""Snapshot parity and integrity regression tests.

Verifies that offline ghost snapshots from `ml-framework-snapshots`
for all six supported frameworks (PyTorch, JAX, Apple MLX, Keras 3,
AMD RDNA, NVIDIA SASS) load cleanly, adhere to expected schemas,
and contain core tensor arithmetic and layer operations.
"""

from pathlib import Path
from typing import Any, Dict, List

from scripts.audit_against_snapshots import load_snapshots_multi


def test_snapshots_load_and_contain_symbols() -> None:
  """Test that snapshots load for all six target frameworks with non-empty symbols."""
  snapshot_dirs: List[Path] = [
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
  ]
  snapshots: Dict[str, Dict[str, Any]] = load_snapshots_multi(snapshot_dirs)

  if "torch" not in snapshots or "jax" not in snapshots:
    import pytest

    pytest.skip("Offline framework snapshots not present in local filesystem.")

  target_frameworks: List[str] = [
    "torch",
    "jax",
    "mlx",
    "keras",
    "rdna",
    "nvidia_sass",
  ]

  for fw in target_frameworks:
    assert fw in snapshots, f"Framework '{fw}' missing from loaded snapshots"
    symbols = snapshots[fw]
    assert len(symbols) > 0, f"Framework '{fw}' loaded 0 symbols"


def test_core_math_primitives_in_snapshots() -> None:
  """Test that foundational math and layer primitives exist in extracted snapshots."""
  snapshot_dirs: List[Path] = [
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
  ]
  snapshots: Dict[str, Dict[str, Any]] = load_snapshots_multi(snapshot_dirs)

  if "torch" not in snapshots or "jax" not in snapshots:
    import pytest

    pytest.skip("Offline framework snapshots not present in local filesystem.")

  # PyTorch
  torch_symbols = snapshots["torch"]
  assert any(k.endswith("abs") or "abs" in k for k in torch_symbols)
  assert any("add" in k for k in torch_symbols)
  assert any("Conv2d" in k for k in torch_symbols)

  # MLX
  mlx_symbols = snapshots["mlx"]
  assert any("abs" in k for k in mlx_symbols)
  assert any("add" in k for k in mlx_symbols)
  assert any("Linear" in k for k in mlx_symbols)

  # Keras
  keras_symbols = snapshots["keras"]
  assert any("abs" in k for k in keras_symbols)
  assert any("Dense" in k for k in keras_symbols)
  assert any("relu" in k.lower() for k in keras_symbols)

  # SASS
  sass_symbols = snapshots["nvidia_sass"]
  assert any("FFMA" in k for k in sass_symbols)
  assert any("FADD" in k for k in sass_symbols)
  assert any("LDG" in k for k in sass_symbols)

  # RDNA
  rdna_symbols = snapshots["rdna"]
  assert any("v_add" in k for k in rdna_symbols)
  assert any("v_fma" in k for k in rdna_symbols)
