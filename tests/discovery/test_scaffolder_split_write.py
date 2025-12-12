"""
Tests for Split-Write Logic in Scaffolder (Distributed Semantics).
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {
        "torch.abs": {"name": "abs", "params": ["x"], "docstring_summary": "Abs"},
        "torch.nn.Linear": {"name": "Linear", "type": "class", "params": ["in", "out"]},
      }
    return {}


@pytest.fixture
def env_paths(tmp_path):
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()
  return sem_dir, snap_dir


def test_scaffold_splits_data(env_paths):
  sem_dir, snap_dir = env_paths

  # Prevent file loading during init for speed and isolation
  with patch("ml_switcheroo.semantics.manager.SemanticsManager._load_knowledge_graph"):
    mgr = SemanticsManager()
    mgr._reverse_index = {}
    mgr.data = {}
    mgr._key_origins = {}

  scaffolder = Scaffolder(semantics=mgr)
  scaffolder.inspector = MockInspector()

  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.discovery.scaffolder.available_frameworks", return_value=["torch"]):
        # Fix: Configure heuristics so Linear is routed to Neural Tier
        mock_adapter = MagicMock()
        mock_adapter.discovery_heuristics = {"neural": [r"\.nn\."]}
        with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
          scaffolder.scaffold(["torch"], sem_dir)

  array_spec = sem_dir / "k_array_api.json"
  assert array_spec.exists()
  array_data = json.loads(array_spec.read_text())

  assert "abs" in array_data
  assert "variants" not in array_data["abs"]

  neural_spec = sem_dir / "k_neural_net.json"
  assert neural_spec.exists()

  torch_snap = snap_dir / "torch_mappings.json"
  assert torch_snap.exists()
  snap_data = json.loads(torch_snap.read_text())

  assert snap_data["__framework__"] == "torch"
  mappings = snap_data["mappings"]
  assert mappings["abs"]["api"] == "torch.abs"


def test_scaffolder_caches_existing_specs(env_paths):
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

  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=snap_dir):
      with patch("ml_switcheroo.discovery.scaffolder.available_frameworks", return_value=["torch"]):
        with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=MagicMock()):
          scaffolder.scaffold(["torch"], sem_dir)

  # Verify persistence
  new_spec = json.loads((sem_dir / "k_array_api.json").read_text())
  assert new_spec["abs"]["description"] == "Manual"

  # Verify mapping
  snap = json.loads((snap_dir / "torch_mappings.json").read_text())
  assert snap["mappings"]["abs"]["api"] == "torch.abs"


def test_static_injection_dataloader_split(env_paths):
  """
  Verifies that Scaffolder injects Abstract Definitions from python defaults,
  BUT does NOT create implementation mappings (variants) because they have been purged.
  """
  sem_dir, snap_dir = env_paths

  with patch("ml_switcheroo.semantics.manager.SemanticsManager._load_knowledge_graph"):
    mgr = SemanticsManager()
    mgr._reverse_index = {}
    mgr.data = {}
    mgr._key_origins = {}

  scaffolder = Scaffolder(semantics=mgr)
  # Ensure no live inspection findings
  scaffolder.inspector.inspect = MagicMock(return_value={})

  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=sem_dir):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=snap_dir):
      scaffolder.scaffold(["torch"], sem_dir)

  # 1. Verify Abstract Spec IS created
  extras_spec = sem_dir / "k_framework_extras.json"
  assert extras_spec.exists()

  extras_data = json.loads(extras_spec.read_text())
  assert "DataLoader" in extras_data
  # It should contain std_args from the python default
  assert "dataset" in extras_data["DataLoader"]["std_args"]
  # It should NOT contain variants
  assert "variants" not in extras_data["DataLoader"]

  # 2. Verify Mapping File is NOT created
  # With variants purged from python code, Scaffolder has nothing to write to the snapshot
  # unless inspecting a live framework, which is mocked to empty here.
  torch_snap = snap_dir / "torch_mappings.json"
  assert not torch_snap.exists()
