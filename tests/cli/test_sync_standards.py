"""
Tests for Sync-Standards CLI Command (Consensus Engine Integration).

Verifies that:
1. The command invokes `collect_api` on registered adapters.
2. It correctly aggregates GhostRefs into the Consensus Engine.
3. It filters non-consensual candidates.
4. It persists discovered standards to JSON.
5. It handles empty categories or missing frameworks gracefully.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.cli import commands
from ml_switcheroo.frameworks.base import StandardCategory, GhostRef
from ml_switcheroo.discovery.consensus import CandidateStandard


@pytest.fixture
def mock_adapters():
  """
  Mocks registry returning two valid adapters (Torch and JAX)
  configured to return compatible GhostRefs for 'LOSS'.
  """
  # Adapter A (Torch)
  torch_adapter = MagicMock()
  # Defines HuberLoss with 'reduction' kwarg
  ref_a = GhostRef(
    name="HuberLoss", api_path="torch.nn.HuberLoss", kind="class", params=[{"name": "reduction", "kind": "kw"}]
  )
  # collect_api returns list of ghosts
  torch_adapter.collect_api.return_value = [ref_a]

  # Adapter B (JAX)
  jax_adapter = MagicMock()
  # Defines Huber with 'reduction' kwarg
  ref_b = GhostRef(name="Huber", api_path="optax.huber", kind="function", params=[{"name": "reduction", "kind": "kw"}])
  jax_adapter.collect_api.return_value = [ref_b]

  def get_adapter_side_effect(name):
    if name == "torch":
      return torch_adapter
    if name == "jax":
      return jax_adapter
    return None

  return get_adapter_side_effect


@pytest.fixture
def mock_persister():
  with patch("ml_switcheroo.cli.commands.SemanticPersister") as mock:
    yield mock


@pytest.fixture
def mock_consensus_engine():
  """Mocks the engine to control clustering output."""
  with patch("ml_switcheroo.cli.commands.ConsensusEngine") as mock:
    # Setup default behavior
    engine = mock.return_value

    # Mock clustering to return a dummy candidate if inputs are provided
    def cluster_side_effect(framework_inputs):
      if not framework_inputs:
        return []
        # Simply return a dummy candidate for 'Huber'
      return [CandidateStandard(name="Huber", variants={}, score=1.0)]

    engine.cluster.side_effect = cluster_side_effect

    # Mock filter to pass through everything for simplicity in this test
    engine.filter_common.side_effect = lambda c, min_support: c

    yield engine


@patch("ml_switcheroo.cli.commands.resolve_semantics_dir")
@patch("ml_switcheroo.cli.commands.get_adapter")
@patch("ml_switcheroo.cli.commands.available_frameworks")
def test_sync_standards_happy_path(
  mock_avail, mock_get_adapter, mock_resolve, mock_adapters, mock_persister, mock_consensus_engine
):
  """
  Scenario:
    - 2 frameworks available (torch, jax).
    - Both return data for 'loss'.
    - Consensus finds 'Huber' match.
  Expectation:
    - SemanticPersister.persist is called with 'Huber' candidate.
  """
  # Setup Mocks
  mock_resolve.return_value = MagicMock()
  mock_avail.return_value = ["torch", "jax"]
  mock_get_adapter.side_effect = mock_adapters

  # Run Command
  ret = commands.handle_sync_standards(
    categories=["loss"],
    frameworks=None,  # All
    dry_run=False,
  )

  assert ret == 0

  # Verify Consensus Engine Calls
  engine = mock_consensus_engine
  engine.cluster.assert_called_once()
  engine.align_signatures.assert_called_once()

  # Verify Persistence
  persister = mock_persister.return_value
  persister.persist.assert_called_once()

  # Check args passed to persist
  args = persister.persist.call_args
  candidates = args[0][0]
  assert len(candidates) == 1
  assert candidates[0].name == "Huber"


@patch("ml_switcheroo.cli.commands.resolve_semantics_dir")
@patch("ml_switcheroo.cli.commands.get_adapter")
@patch("ml_switcheroo.cli.commands.available_frameworks")
def test_sync_standards_not_enough_data(mock_avail, mock_get_adapter, mock_resolve, mock_persister):
  """
  Scenario: Only 1 framework returns data.
  Expectation: Consensus skipped for that category.
  """
  mock_resolve.return_value = MagicMock()
  mock_avail.return_value = ["torch"]

  # Adapter returns data, but since only 1 FW, len(framework_inputs) == 1
  mock_adapter = MagicMock()
  mock_adapter.collect_api.return_value = [GhostRef(name="A", api_path="a", kind="f")]
  mock_get_adapter.return_value = mock_adapter

  commands.handle_sync_standards(["loss"], None, False)

  # Persister should NOT be called because < 2 frameworks input
  mock_persister.return_value.persist.assert_not_called()


def test_invalid_category_handled_gracefully():
  """
  Scenario: User passes invalid category string 'magic'.
  Expectation: Log warning, iterate only valid ones.
  """
  # Patch internal calls via with context if we want integration,
  # or check logic directly.
  # Here we just verify it doesn't crash.

  with patch("ml_switcheroo.cli.commands.available_frameworks", return_value=[]):
    ret = commands.handle_sync_standards(
      categories=["magic"],  # Invalid
      frameworks=[],
      dry_run=True,
    )

    # Should return 1 (error) as no valid categories found
  assert ret == 1
