"""
Tests for Spec-Driven Neural Scaffolding (Split Write).
"""

import json
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class MockInspector:
  def inspect(self, fw):
    # Handle modules
    return {
      "torch.nn.ReLU": {"name": "ReLU", "type": "class", "params": []},
      "torch.custom.Unknown": {"name": "UnknownLayer", "type": "class", "params": []},
    }


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {"Relu": {"std_args": ["x"]}}
    self._key_origins = {"Relu": SemanticTier.NEURAL.value}


def test_spec_driven_categorization(tmp_path):
  scaffolder = Scaffolder(semantics=MockSemantics())
  scaffolder.inspector = MockInspector()

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  mock_adapter = MagicMock()
  mock_adapter.search_modules = None

  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    # Fix: Patch version detection to ensure deterministic filename
    with patch("ml_switcheroo.discovery.scaffolder.importlib.metadata.version", return_value="latest"):
      with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
        with patch.dict("sys.modules", {"torch": MagicMock(__version__="latest")}):
          scaffolder.scaffold(["torch"], root_dir=tmp_path)

  neural_spec = json.loads((sem_dir / "k_neural_net.json").read_text())
  assert "Relu" in neural_spec

  torch_map = json.loads((snap_dir / "torch_vlatest_map.json").read_text())
  assert torch_map["mappings"]["Relu"]["api"] == "torch.nn.ReLU"


def test_heuristic_fallback_dynamic(tmp_path):
  semantics = SemanticsManager()
  semantics.data = {}
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.inspector = MockInspector()

  sem_dir = tmp_path / "semantics"
  snap_dir = tmp_path / "snapshots"
  sem_dir.mkdir()
  snap_dir.mkdir()

  mock_adapter = MagicMock()
  mock_adapter.discovery_heuristics = {"neural": [r"\\.custom\\."]}
  mock_adapter.search_modules = None

  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["torch"]):
    with patch("ml_switcheroo.discovery.scaffolder.importlib.metadata.version", return_value="latest"):
      with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
        with patch.dict("sys.modules", {"torch": MagicMock(__version__="latest")}):
          scaffolder.scaffold(["torch"], root_dir=tmp_path)

  torch_map = json.loads((snap_dir / "torch_vlatest_map.json").read_text())
  assert "UnknownLayer" in torch_map["mappings"]
  assert torch_map["mappings"]["UnknownLayer"]["api"] == "torch.custom.Unknown"
