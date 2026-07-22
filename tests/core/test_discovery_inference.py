"""Test suite for the Discovery Inference module."""

import types
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.discovery import SimulatedReflection


def mock_module_with_members(name: str, members: list):
  """Provides a mock module with members for testing."""
  mod = types.ModuleType(name)
  for m in members:
    setattr(mod, m, MagicMock())
  return mod


def get_mock_adapter(modules=None):
  """Gets mock adapter."""
  adp = MagicMock()
  adp.search_modules = modules or ["mock_fw", "mock_fw.nn"]
  return adp


def test_exact_match():
  """Verifies the behavior of exact match."""
  adapter = get_mock_adapter()
  mod_root = mock_module_with_members("mock_fw", ["foo"])
  mod_nn = mock_module_with_members("mock_fw.nn", ["LogSoftmax", "ReLU"])

  def import_side_effect(name):
    """Helper to import side effect."""
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
  """Verifies the behavior of normalized match."""
  adapter = get_mock_adapter()
  mod_nn = mock_module_with_members("mock_fw.nn", ["log_softmax"])
  mod_root = mock_module_with_members("mock_fw", [])

  def side_effect(name):
    """Helper to side effect."""
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
  """Verifies the behavior of fuzzy match."""
  adapter = get_mock_adapter(modules=["mock_fw"])
  mod = mock_module_with_members("mock_fw", ["softmax_v2"])
  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mod):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("softmax")
  assert result == "mock_fw.softmax_v2"


def test_no_match_returns_none():
  """Verifies the behavior of no match returns none."""
  adapter = get_mock_adapter()
  mod = mock_module_with_members("mock_fw", ["nothing_relevant"])
  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mod):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("SpecificOp")
  assert result is None


def test_missing_adapter_fallback():
  """Verifies the behavior of missing adapter fallback."""
  mod = mock_module_with_members("ghost_fw", ["Op"])
  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=None):
    with patch("importlib.import_module", return_value=mod) as mock_import:
      reflector = SimulatedReflection("ghost_fw")
      assert reflector.search_modules == ["ghost_fw"]
      result = reflector.discover("Op")
      assert result == "ghost_fw.Op"
      mock_import.assert_called_with("ghost_fw")


def test_import_error_handled_gracefully():
  """Verifies the behavior of import correctly handling an error handled gracefully."""
  adapter = get_mock_adapter(modules=["bad_mod", "good_mod"])
  mod_good = mock_module_with_members("good_mod", ["Target"])

  def side_effect(name):
    """Helper to side effect."""
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


def test_fuzzy_match_import_error():
  """Verifies the behavior of fuzzy match import correctly handling an error."""
  adapter = get_mock_adapter(modules=["bad_mod", "good_mod"])
  mod_good = mock_module_with_members("good_mod", ["Target_v2"])

  def side_effect(name):
    """Helper to side effect."""
    if name == "bad_mod":
      raise ImportError("Broken")
    if name == "good_mod":
      return mod_good
    return None

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=side_effect):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("Target")
  assert result == "good_mod.Target_v2"


def test_fuzzy_match_no_candidates():
  """Verifies the behavior of fuzzy match no candidates."""
  adapter = get_mock_adapter(modules=["bad_mod"])

  def side_effect(name):
    """Helper to side effect."""
    raise ImportError("Broken")

  with patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=side_effect):
      reflector = SimulatedReflection("test_fw")
      result = reflector.discover("Target")
  assert result is None
