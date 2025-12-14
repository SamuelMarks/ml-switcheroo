"""
Tests for Scaffolder Dtype support (Split Write).
"""

import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


class MockAttributeInspector:
  def inspect(self, fw):
    if fw == "torch":
      return {"torch.float32": {"name": "float32", "type": "attribute", "params": []}}
    return {}


def test_scaffolder_propagates_type_field(tmp_path):
  clean_semantics = SemanticsManager()
  clean_semantics.data = {}
  # Fix: Initialize missing attributes
  clean_semantics._key_origins = {}

  scaffolder = Scaffolder(semantics=clean_semantics)
  scaffolder.inspector = MockAttributeInspector()

  # Use subdir structure to match logical expectation of sibling snapshots
  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  # Patch where it is DEFINED to affect all imports, since Scaffolder uses it internally
  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    # Patch adapter to avoid real torch lookup
    with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=MagicMock()):
      # Patch importlib.metadata.version to return 'latest' directly for deterministic filenames
      with patch("importlib.metadata.version", return_value="latest"):
        # Also need to patch torch.__version__ if torch is importable
        with patch("ml_switcheroo.discovery.scaffolder.importlib.metadata.version", return_value="latest"):
          # Force version inside the logic block if try/except falls through
          with patch.dict("sys.modules", {"torch": MagicMock(__version__="latest")}):
            scaffolder.scaffold(["torch"], root_dir=tmp_path)

  # Check Spec (might be in extras if type not matched, but let's check both)
  spec_path = sem_dir / "k_array_api.json"
  if not spec_path.exists():
    spec_path = sem_dir / "k_framework_extras.json"

  spec = json.loads(spec_path.read_text())
  assert spec["float32"]["type"] == "attribute"

  # Check Mapping
  # Now guaranteed to be vlatest because we mocked the version retrieval
  snap = json.loads((snap_dir / "torch_vlatest_map.json").read_text())
  assert snap["mappings"]["float32"]["api"] == "torch.float32"
