import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def manager():
  return SemanticsManager()


def test_load_semantics(manager):
  """Verify definitions were loaded from JSON."""
  data = manager.get_known_apis()
  assert len(data) > 0
  assert "abs" in data


def test_resolve_torch_sum(manager):
  """
  Test reverse lookup logic.
  """
  # 1. Define Mock Data
  mock_data = {"mock_op": {"variants": {"torch": {"api": "torch.mock_op"}}}}

  # 2. Inject Data directly (simulating a load)
  # The new manager builds index from self.data, not argument
  manager.data = mock_data

  # 3. Trigger Index Build
  manager._build_index()

  # 4. Test lookup
  # get_definition returns (std_name, full_definition_dict)
  result = manager.get_definition("torch.mock_op")

  # Verify we got a tuple and the name is correct
  assert isinstance(result, tuple)
  assert result[0] == "mock_op"
  assert result[1]["variants"]["torch"]["api"] == "torch.mock_op"


def test_unknown_api_returns_none(manager):
  result = manager.get_definition("torch.non_existent_function")
  assert result is None
