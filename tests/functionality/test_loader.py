"""Test suite for the Loader module."""

import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def manager():
  """Provides a mock manager for testing."""
  return SemanticsManager()


def test_load_semantics(manager):
  """Loads semantics."""
  data = manager.get_known_apis()
  assert len(data) > 0
  assert "abs" in data


def test_resolve_torch_sum(manager):
  """Resolves PyTorch sum."""
  mock_data = {"mock_op": {"variants": {"torch": {"api": "torch.mock_op"}}}}
  manager.data = mock_data
  manager._build_index()
  result = manager.get_definition("torch.mock_op")
  assert isinstance(result, tuple)
  assert result[0] == "mock_op"
  assert result[1]["variants"]["torch"]["api"] == "torch.mock_op"


def test_unknown_api_returns_none(manager):
  """Verifies the behavior of unknown API returns none."""
  result = manager.get_definition("torch.non_existent_function")
  assert result is None
