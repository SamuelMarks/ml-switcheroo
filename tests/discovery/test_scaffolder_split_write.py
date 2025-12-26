"""
Tests for Split-Write Logic in Scaffolder (Distributed Semantics).
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, fw, **kwargs):
    # Robust return for any submodule calls
    return {
      "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Abs"},
      "torch.nn.Linear": {"name": "Linear", "type": "class", "params": ["in", "out"]},
    }


@pytest.fixture
def env_paths(tmp_path):
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()
  return sem_dir, snap_dir


def test_scaffold_splits_data(env_paths, tmp_path):
  sem_dir, snap_dir = env_paths

  # Prevent file loading during init for speed and isolation
  with patch("ml_switcheroo.semantics.manager.SemanticsManager._load_knowledge_graph"):
    mgr = SemanticsManager()
    mgr._reverse_index = {}
    mgr.data = {}
    mgr._key_origins = {}

  # Create Scaffolder instance
  scaffolder = Scaffolder(semantics=mgr)
  scaffolder.inspector = MockInspector()

  # Helper to force bypass regex complexity in test environment
  # We stub internal decision logic to ensure routing works
  scaffolder._is_structurally_neural = MagicMock(side_effect=lambda path, kind: "Linear" in path)

  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    # Fix: Patch version detection to ensure deterministic filename
    with patch("ml_switcheroo.discovery.scaffolder.importlib.metadata.version", return_value="latest"):
      # Fix: Configure heuristics
      mock_adapter = MagicMock()
      mock_adapter.discovery_heuristics = {"neural": [r"\\.nn\\."]}
      # Explicitly disable search modules traversal
      mock_adapter.search_modules = None
      mock_adapter.unsafe_submodules = set()

      with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
        with patch.dict("sys.modules", {"torch": MagicMock(__version__="latest")}):
          scaffolder.scaffold(["torch"], root_dir=tmp_path)

  array_spec = sem_dir / "k_array_api.json"
  if not array_spec.exists():
    # Fallback check
    array_spec = sem_dir / "k_framework_extras.json"

  assert array_spec.exists()
  array_data = json.loads(array_spec.read_text())

  assert "abs" in array_data
  assert "variants" not in array_data["abs"]

  # Verify Neural Spec creation (Focus of the test)
  neural_spec = sem_dir / "k_neural_net.json"
  assert neural_spec.exists()

  torch_snap = snap_dir / "torch_vlatest_map.json"
  assert torch_snap.exists()
  snap_data = json.loads(torch_snap.read_text())

  assert snap_data["__framework__"] == "torch"
  mappings = snap_data["mappings"]
  assert mappings["abs"]["api"] == "torch.abs"
  assert mappings["Linear"]["api"] == "torch.nn.Linear"


def test_scaffolder_caches_existing_specs(env_paths, tmp_path):
  sem_dir, snap_dir = env_paths

  existing_spec = {"abs": {"description": "Manual", "std_args": ["x"]}}
  (sem_dir / "k_array_api.json").write_text(json.dumps(existing_spec))

  with patch("ml_switcheroo.semantics.manager.SemanticsManager._load_knowledge_graph"):
    mgr = SemanticsManager()
    mgr._reverse_index = {}
    mgr.data = existing_spec
    # Fix: Populate origins
    mgr._key_origins = {"abs": "array"}

  scaffolder = Scaffolder(semantics=mgr)
  scaffolder.inspector = MockInspector()

  mock_adapter = MagicMock()
  mock_adapter.search_modules = None
  mock_adapter.unsafe_submodules = set()

  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    with patch("ml_switcheroo.discovery.scaffolder.importlib.metadata.version", return_value="latest"):
      with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
        with patch.dict("sys.modules", {"torch": MagicMock(__version__="latest")}):
          scaffolder.scaffold(["torch"], root_dir=tmp_path)

  # Verify persistence
  new_spec = json.loads((sem_dir / "k_array_api.json").read_text())
  assert new_spec["abs"]["description"] == "Manual"

  # Verify mapping
  snap = json.loads((snap_dir / "torch_vlatest_map.json").read_text())
  assert snap["mappings"]["abs"]["api"] == "torch.abs"
