"""Test suite for the Manager Overlay Loading module."""

import json
import typing
from pathlib import Path
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_root_tree(tmp_path: Path) -> Path:
  """Provides a mock root tree for testing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.

  Returns:
      Path: Tmp path.
  """
  semantics_dir: Path = tmp_path / "semantics"
  semantics_dir.mkdir()
  snapshots_dir: Path = tmp_path / "snapshots"
  snapshots_dir.mkdir()
  spec_content = {
    "Abs": {"description": "Calculate absolute value", "std_args": ["x"]},
    "Add": {"description": "Addition", "std_args": ["a", "b"]},
  }
  import yaml

  odl_dir: Path = semantics_dir / "odl"
  odl_dir.mkdir()
  (odl_dir / "add.yaml").write_text(yaml.dump({"Add": spec_content["Add"]}))
  (odl_dir / "abs.yaml").write_text(yaml.dump({"Abs": spec_content["Abs"]}))
  torch_map = {
    "__framework__": "torch",
    "mappings": {"Abs": {"api": "torch.abs"}, "Add": {"api": "torch.add"}, "custom_op": {"api": "torch.special"}},
  }
  (snapshots_dir / "torch_vlatest_map.json").write_text(json.dumps(torch_map))
  jax_map = {
    "__framework__": "jax",
    "mappings": {"Abs": {"api": "jax.numpy.abs"}, "Add": {"api": "jax.numpy.add"}},
  }
  (snapshots_dir / "jax_vlatest_map.json").write_text(json.dumps(jax_map))
  return semantics_dir


@pytest.fixture
def manager(mock_root_tree: Path) -> typing.Generator[SemanticsManager, None, None]:
  """Provides a mock manager for testing.

  Args:
      mock_root_tree (Path): Mock root tree pytest fixture.

  Yields:
      SemanticsManager: Mock SemanticsManager.
  """
  with (
    patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=mock_root_tree),
    patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=mock_root_tree.parent / "snapshots"),
    patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]),
  ):
    yield SemanticsManager()


def test_overlay_merging_logic(manager: SemanticsManager) -> None:
  """Verify overlay merging logic produces the expected state.

  Args:
      manager (SemanticsManager): Manager fixture.
  """
  assert "Abs" in manager.data
  entry = manager.data["Abs"]
  variants = entry["variants"]
  assert isinstance(variants, dict)
  assert "torch" in variants
  torch_var = variants["torch"]
  assert isinstance(torch_var, dict)
  assert torch_var["api"] == "torch.abs"
  assert "jax" in variants
  jax_var = variants["jax"]
  assert isinstance(jax_var, dict)
  assert jax_var["api"] == "jax.numpy.abs"


def test_overlay_missing_op_handling(manager: SemanticsManager) -> None:
  """Verify overlay missing operation handling produces the expected state.

  Args:
      manager (SemanticsManager): Manager fixture.
  """
  assert "custom_op" in manager.data
  entry = manager.data["custom_op"]
  assert manager._key_origins["custom_op"] == SemanticTier.EXTRAS.value
  variants = entry["variants"]
  assert isinstance(variants, dict)
  torch_var = variants["torch"]
  assert isinstance(torch_var, dict)
  assert torch_var["api"] == "torch.special"
  desc = entry["description"]
  assert isinstance(desc, str)
  assert "Auto-generated" in desc


def test_filename_framework_inference(tmp_path: Path) -> None:
  """Verify `dumps` emits `Sin` for filename framework inference.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  sem_dir: Path = tmp_path / "semantics"
  sem_dir.mkdir()
  snap_dir: Path = tmp_path / "snapshots"
  snap_dir.mkdir()
  (sem_dir / "k_math.json").write_text(json.dumps({"Sin": {}}))
  numpy_map = {"mappings": {"Sin": {"api": "numpy.sin"}}}
  (snap_dir / "numpy_vlatest_map.json").write_text(json.dumps(numpy_map))
  with (
    patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem_dir),
    patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap_dir),
    patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]),
  ):
    mgr: SemanticsManager = SemanticsManager()
    mgr._reverse_index = {}
    assert "Sin" in mgr.data
    sin_variants = mgr.data["Sin"]["variants"]
    assert isinstance(sin_variants, dict)
    assert "numpy" in sin_variants
    numpy_var = sin_variants["numpy"]
    assert isinstance(numpy_var, dict)
    assert numpy_var["api"] == "numpy.sin"


def test_reverse_index_integrity(manager: SemanticsManager) -> None:
  """Verify `manager.get_definition` handles reverse index integrity as expected.

  Args:
      manager (SemanticsManager): Manager fixture.
  """
  lookup = manager.get_definition("torch.abs")
  assert lookup is not None
  (abstract_id, data) = lookup
  assert abstract_id == "Abs"
  variants = data["variants"]
  assert isinstance(variants, dict)
  torch_var = variants["torch"]
  assert isinstance(torch_var, dict)
  assert torch_var["api"] == "torch.abs"
