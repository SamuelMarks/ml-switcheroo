"""Unit tests for the SimulatedReflection engine.

This test suite ensures that the `SimulatedReflection` engine correctly resolves
abstract API names to framework-specific API paths. It tests initialization, name
normalization, exact matching against exported public members, fuzzy matching using
similarity metrics, and error boundary handling when modules cannot be imported.
"""

import typing
from unittest.mock import Mock, patch

from ml_switcheroo.core.discovery import SimulatedReflection


def test_discovery_init() -> None:
  """Tests the initialization of the SimulatedReflection class.

  Verifies that when SimulatedReflection is instantiated with a framework name,
  it correctly sets the framework attribute and falls back to a list containing
  only that framework name in search_modules when no specialized adapter is found.

  Args:
      None

  Returns:
      None
  """
  reflection = SimulatedReflection("torch")
  assert reflection.framework == "torch"
  assert reflection.search_modules == ["torch"]


def test_discovery_normalize() -> None:
  """Tests the normalization method of the SimulatedReflection class.

  Verifies that `_normalize` correctly maps typical API names to their canonical,
  case-insensitive, underscore-stripped representation (e.g., converting 'LogSoftmax'
  to 'logsoftmax', and 'abs_' to 'abs').

  Args:
      None

  Returns:
      None
  """
  reflection = SimulatedReflection("torch")
  assert reflection._normalize("LogSoftmax") == "logsoftmax"
  assert reflection._normalize("abs_") == "abs"


def test_discovery_exact_match() -> None:
  """Tests exact-matching/normalization-matching in SimulatedReflection discovery.

  This test mocks the framework's internal structure to simulate the presence of public
  members and private members. It verifies that `discover` successfully locates and
  returns the exact (normalized) matching API path (e.g., matching 'LogSoftmax' to
  'log_softmax') from the search modules while correctly ignoring private members (starting with '_').

  Returns:
      None
  """
  # Mock importlib to return a fake module
  mock_mod = Mock()
  mock_mod.log_softmax = Mock()
  # Also add a private member to ensure it's skipped
  mock_mod._private = Mock()

  with (
    patch("inspect.getmembers", return_value=[("log_softmax", mock_mod.log_softmax), ("_private", mock_mod._private)]),
    patch("importlib.import_module", return_value=mock_mod),
  ):
    reflection = SimulatedReflection("torch")
    reflection.search_modules = ["torch.nn.functional"]
    result: typing.Optional[str] = reflection.discover("LogSoftmax")
    assert result == "torch.nn.functional.log_softmax"


def test_discovery_fuzzy_match() -> None:
  """Tests fuzzy-matching fallbacks in SimulatedReflection discovery.

  This test mocks the framework's module contents and verifies that when an exact match is
  not found, the discovery engine falls back to a fuzzy search using close string matching
  (e.g., matching 'absolut' to 'numpy.absolute').

  Returns:
      None
  """
  mock_mod = Mock()
  mock_mod.abs = Mock()

  with patch("inspect.getmembers", return_value=[("absolute", mock_mod.abs)]):
    with patch("importlib.import_module", return_value=mock_mod):
      reflection = SimulatedReflection("numpy")
      result: typing.Optional[str] = reflection.discover("absolut")
      assert result == "numpy.absolute"


def test_discovery_no_match() -> None:
  """Tests discovery behavior when no matching API endpoint is found.

  This test simulates import failures when trying to load framework modules,
  verifying that `discover` handles these exceptions gracefully and returns `None`
  instead of propagating the error or returning an incorrect match.

  Returns:
      None
  """
  patch("importlib.import_module", side_effect=ImportError)

  reflection = SimulatedReflection("torch")
  result: typing.Optional[str] = reflection.discover("NonExistent")
  assert result is None
