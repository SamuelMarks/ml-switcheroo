"""
Tests for Layer Discovery Logic (src/ml_switcheroo/discovery/layers.py).

Verifies that the Bot:
1. Iterates over registered frameworks.
2. Collects API surfaces via adapters.
3. Groups them via Consensus Engine.
4. Persists the result.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.discovery.layers import LayerDiscoveryBot
from ml_switcheroo.core.ghost import GhostRef, GhostParam


@pytest.fixture
def mock_adapters():
  """
  Creates mock adapters for 'torch' and 'flax' that return predictable GhostRefs.
  """
  # Torch Adapter Mock
  torch_adp = MagicMock()
  # When collect_api is called, return Linear/Conv2d
  torch_linear = GhostRef(name="Linear", api_path="torch.nn.Linear", kind="class", params=[])
  torch_conv = GhostRef(name="Conv2d", api_path="torch.nn.Conv2d", kind="class", params=[])
  torch_relu = GhostRef(name="ReLU", api_path="torch.nn.ReLU", kind="class", params=[])

  torch_adp.collect_api.return_value = [torch_linear, torch_conv, torch_relu]

  # Flax Adapter Mock
  flax_adp = MagicMock()
  flax_linear = GhostRef(name="Linear", api_path="flax.nnx.Linear", kind="class", params=[])
  # Note: Flax might not have Conv2d with exact same name in this mock, checking consensus
  flax_relu = GhostRef(name="relu", api_path="flax.nnx.relu", kind="function", params=[])

  flax_adp.collect_api.return_value = [flax_linear, flax_relu]

  return {"torch": torch_adp, "flax_nnx": flax_adp}


def test_e2e_discovery_execution(tmp_path, mock_adapters):
  """
  Simulates full run writing to tmp_path by patching infrastructure.
  """
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()

  bot = LayerDiscoveryBot()

  # 1. Patch Path Resolution
  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=sem_dir):
    # 2. Patch Framework Discovery
    # avaiable_frameworks returns keys
    with patch("ml_switcheroo.discovery.layers.available_frameworks", return_value=["torch", "flax_nnx"]):
      # 3. Patch Adapter Retrieval
      def side_effect(name):
        return mock_adapters.get(name)

      with patch("ml_switcheroo.discovery.layers.get_adapter", side_effect=side_effect):
        # Run Discovery
        count = bot.run(dry_run=False)

        # We expect at least 'Linear' and 'ReLU'/'relu' to match
        assert count > 0

  outfile = sem_dir / "k_discovered.json"
  assert outfile.exists()
  data = json.loads(outfile.read_text())

  # Both modules define Linear, so it should be present in consensus
  assert "Linear" in data

  # 'ReLU' (Torch) and 'relu' (Flax) should cluster to 'Relu' or 'relu'
  # depending on casing logic in ConsensusEngine (usually normalizes to lowercase key)
  # The actual output key might be 'Relu' (Capitalized by CandidateStandard default).
  # Let's check for existence of the concept.
  assert "Relu" in data or "relu" in data


def test_dry_run_does_not_write(tmp_path, mock_adapters):
  """Verify dry run logic."""
  bot = LayerDiscoveryBot()

  with patch("ml_switcheroo.semantics.paths.resolve_semantics_dir", return_value=tmp_path):
    with patch("ml_switcheroo.discovery.layers.available_frameworks", return_value=["torch", "flax_nnx"]):
      with patch("ml_switcheroo.discovery.layers.get_adapter", side_effect=lambda n: mock_adapters.get(n)):
        bot.run(dry_run=True)

  outfile = tmp_path / "k_discovered.json"
  assert not outfile.exists()


def test_insufficient_frameworks_aborts(tmp_path):
  """Verify scripts stops if < 2 frameworks found."""
  bot = LayerDiscoveryBot()

  # Only torch available
  mock_single_adapter = MagicMock()
  mock_single_adapter.collect_api.return_value = [GhostRef(name="A", api_path="a", kind="c")]

  with patch("ml_switcheroo.discovery.layers.available_frameworks", return_value=["torch"]):
    with patch("ml_switcheroo.discovery.layers.get_adapter", return_value=mock_single_adapter):
      count = bot.run(dry_run=True)

  assert count == 0
