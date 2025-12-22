"""
Tests for Core Discovery and Inference Logic.

Verifies:
1. SimulatedReflection finds exact matches.
2. SimulatedReflection finds fuzzy matches.
3. Behavior when module is missing or import fails.
4. Behavior when no match is found.
"""

import types
import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.discovery import SimulatedReflection


def mock_module_with_members(name: str, members: list):
  """
  Helper to create a real ModuleType with MagicMock members.
  Allows inspect.getmembers to work correctly.
  """
  mod = types.ModuleType(name)
  for m in members:
    setattr(mod, m, MagicMock())
  return mod


def get_mock_adapter(modules=None):
  """
  Helper to create a mock adapter with search_modules property.
  """
  adp = MagicMock()
  adp.search_modules = modules or ["mock_fw", "mock_fw.nn"]
  return adp


def test_exact_match():
  """
  Scenario: 'LogSoftmax' requested. 'mock_fw.nn' contains 'LogSoftmax' (Exact).
  Expect: 'mock_fw.nn.LogSoftmax'.
  """
  adapter = get_mock_adapter()

  # Setup Modules
  mod_root = mock_module_with_members("mock_fw", ["foo"])
  mod_nn = mock_module_with_members("mock_fw.nn", ["LogSoftmax", "ReLU"])

  def import_side_effect(name):
    if name == "mock_fw":
      return mod_root
    if name == "mock_fw.nn":
      return mod_nn
    raise ImportError(f"No mock for {name}")

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=import_side_effect):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("LogSoftmax")

  assert result == "mock_fw.nn.LogSoftmax"


def test_normalized_match():
  """
  Scenario: 'LogSoftmax' requested. 'mock_fw.nn' contains 'log_softmax'.
  Expect: 'mock_fw.nn.log_softmax' (Normalization matches snake_case vs CamelCase).
  """
  adapter = get_mock_adapter()

  mod_nn = mock_module_with_members("mock_fw.nn", ["log_softmax"])
  mod_root = mock_module_with_members("mock_fw", [])

  def side_effect(name):
    if name == "mock_fw.nn":
      return mod_nn
    if name == "mock_fw":
      return mod_root
    raise ImportError(name)

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=side_effect):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("LogSoftmax")

  assert result == "mock_fw.nn.log_softmax"


def test_fuzzy_match():
  """
  Scenario: 'softmax' requested. 'mock_fw' contains 'softmax_v2'.
  Expect: 'mock_fw.softmax_v2' (Closest string match fallback).
  """
  adapter = get_mock_adapter(modules=["mock_fw"])
  mod = mock_module_with_members("mock_fw", ["softmax_v2"])

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mod):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("softmax")

  assert result == "mock_fw.softmax_v2"


def test_no_match_returns_none():
  """
  Scenario: Operation not found in any search module.
  Expect: None.
  """
  adapter = get_mock_adapter()
  mod = mock_module_with_members("mock_fw", ["nothing_relevant"])

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mod):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("SpecificOp")

  assert result is None


def test_missing_adapter_fallback():
  """
  Scenario: get_adapter returns None for the framework.
  Expect: System falls back to using [framework_name] as the search list.
  """
  mod = mock_module_with_members("ghost_fw", ["Op"])

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=None):
    with patch("importlib.import_module", return_value=mod) as mock_import:
      reflector = SimulatedReflection("ghost_fw")

      # Verify initialized defaults
      assert reflector.search_modules == ["ghost_fw"]

      # Run discovery
      result = reflector.discover("Op")

      assert result == "ghost_fw.Op"
      mock_import.assert_called_with("ghost_fw")


def test_import_error_handled_gracefully():
  """
  Scenario: One of the search modules fails to import.
  Expect: Continues to next module without crashing.
  """
  adapter = get_mock_adapter(modules=["bad_mod", "good_mod"])

  mod_good = mock_module_with_members("good_mod", ["Target"])

  def side_effect(name):
    if name == "bad_mod":
      raise ImportError("Broken")
    if name == "good_mod":
      return mod_good
    return None

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=side_effect):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("Target")

  assert result == "good_mod.Target"
