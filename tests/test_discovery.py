"""Test suite for discovery."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.core.discovery import SimulatedReflection


class MockAdapter:
  """Test helper class MockAdapter."""

  def __init__(self, search_modules: list[str] | None = None) -> None:
    """Initialize test helper instance.

    Args:
        search_modules: Test argument or fixture.
    """
    if search_modules is not None:
      self.search_modules: list[str] = search_modules


@patch("ml_switcheroo.core.discovery.get_adapter")
def test_simulated_reflection_init(mock_get_adapter: MagicMock) -> None:
  """Verify simulated reflection init behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["test_mod1", "test_mod2"])
  ref: SimulatedReflection = SimulatedReflection("test_fw")
  assert ref.search_modules == ["test_mod1", "test_mod2"]


@patch("ml_switcheroo.core.discovery.get_adapter")
def test_simulated_reflection_init_no_search_modules(mock_get_adapter: MagicMock) -> None:
  """Verify simulated reflection init no search modules behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter()
  ref: SimulatedReflection = SimulatedReflection("test_fw")
  assert ref.search_modules == ["test_fw"]


@patch("ml_switcheroo.core.discovery.get_adapter")
def test_simulated_reflection_init_no_adapter(mock_get_adapter: MagicMock) -> None:
  """Verify simulated reflection init no adapter behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
  """
  mock_get_adapter.return_value = None
  ref: SimulatedReflection = SimulatedReflection("test_fw")
  assert ref.search_modules == ["test_fw"]


@patch("ml_switcheroo.core.discovery.importlib.import_module")
@patch("ml_switcheroo.core.discovery.inspect.getmembers")
@patch("ml_switcheroo.core.discovery.get_adapter")
def test_discovery_exact_match(mock_get_adapter: MagicMock, mock_getmembers: MagicMock, mock_import: MagicMock) -> None:
  """Verify discovery exact match behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_getmembers: Test argument or fixture.
      mock_import: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["modA"])
  mock_import.return_value = MagicMock()
  mock_getmembers.return_value = [("_private", None), ("LogSoftmax", None), ("log_softmax", None)]

  ref: SimulatedReflection = SimulatedReflection("test_fw")
  result: str | None = ref.discover("Log_Softmax")

  assert result in ["modA.LogSoftmax", "modA.log_softmax"]


@patch("ml_switcheroo.core.discovery.importlib.import_module")
@patch("ml_switcheroo.core.discovery.inspect.getmembers")
@patch("ml_switcheroo.core.discovery.get_adapter")
def test_discovery_fuzzy_match(mock_get_adapter: MagicMock, mock_getmembers: MagicMock, mock_import: MagicMock) -> None:
  """Verify discovery fuzzy match behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_getmembers: Test argument or fixture.
      mock_import: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["modA"])
  mock_import.return_value = MagicMock()
  # It will not exactly match "log_softmax" normalized because target is "some_other_op"
  # Wait, fuzzy match needs the exact match to fail.
  # Target: "logsoftmax"
  # Member: "log_soft_mx" (normalized: "logsoftmx")
  mock_getmembers.return_value = [("log_soft_mx", None)]

  ref: SimulatedReflection = SimulatedReflection("test_fw")
  result: str | None = ref.discover("logsoftmax")

  assert result == "modA.log_soft_mx"


@patch("ml_switcheroo.core.discovery.importlib.import_module")
@patch("ml_switcheroo.core.discovery.get_adapter")
def test_discovery_import_error(mock_get_adapter: MagicMock, mock_import: MagicMock) -> None:
  """Verify discovery import error behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_import: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["modA"])
  mock_import.side_effect = ImportError("module not found")

  ref: SimulatedReflection = SimulatedReflection("test_fw")
  result: str | None = ref.discover("op")
  assert result is None


@patch("ml_switcheroo.core.discovery.importlib.import_module")
@patch("ml_switcheroo.core.discovery.inspect.getmembers")
@patch("ml_switcheroo.core.discovery.get_adapter")
def test_discovery_fuzzy_import_error(
  mock_get_adapter: MagicMock, mock_getmembers: MagicMock, mock_import: MagicMock
) -> None:
  """Verify discovery fuzzy import error behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_getmembers: Test argument or fixture.
      mock_import: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["modA", "modB"])

  def side_effect(name: str) -> MagicMock:
    """Execute side effect helper logic.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "modA":
      raise ImportError()
    return MagicMock()

  mock_import.side_effect = side_effect

  mock_getmembers.return_value = [("log_soft_mx", None)]

  ref: SimulatedReflection = SimulatedReflection("test_fw")
  result: str | None = ref.discover("logsoftmax")
  assert result == "modB.log_soft_mx"


@patch("ml_switcheroo.core.discovery.importlib.import_module")
@patch("ml_switcheroo.core.discovery.inspect.getmembers")
@patch("ml_switcheroo.core.discovery.get_adapter")
def test_discovery_no_fuzzy_match(
  mock_get_adapter: MagicMock, mock_getmembers: MagicMock, mock_import: MagicMock
) -> None:
  """Verify discovery no fuzzy match behavior.

  Args:
      mock_get_adapter: Test argument or fixture.
      mock_getmembers: Test argument or fixture.
      mock_import: Test argument or fixture.
  """
  mock_get_adapter.return_value = MockAdapter(search_modules=["modA"])
  mock_import.return_value = MagicMock()
  mock_getmembers.return_value = [("completely_different", None)]

  ref: SimulatedReflection = SimulatedReflection("test_fw")
  result: str | None = ref.discover("logsoftmax")
  assert result is None
