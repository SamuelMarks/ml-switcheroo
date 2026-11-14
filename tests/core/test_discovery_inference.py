"""Test suite for the Discovery Inference module."""

import types
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.core.discovery import SimulatedReflection


def mock_module_with_members(monkeypatch: pytest.MonkeyPatch, name: str, members: list[str]) -> types.ModuleType:
  """Build a mock ``types.ModuleType`` populated with scoped ``MagicMock`` members.

  Args:
      monkeypatch: Pytest monkeypatch fixture used for scoped attribute assignment.
      name: Module identifier string.
      members: Member attribute names to attach to the module.

  Returns:
      Populated ``types.ModuleType`` instance.
  """
  mod = types.ModuleType(name)
  for m in members:
    monkeypatch.setattr(mod, m, MagicMock(), raising=False)
  return mod


def get_mock_adapter(modules: list[str] | None = None) -> MagicMock:
  """Gets mock adapter.

  Args:
      modules: Input `modules` parameter value (`list[str] | None`).

  Returns:
      Resulting `MagicMock` from get mock adapter.
  """
  adp = MagicMock()
  adp.search_modules = modules or ["mock_fw", "mock_fw.nn"]
  return adp


def test_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify exact name matching discovers the target symbol in submodules.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter()
  mod_root = mock_module_with_members(monkeypatch, "mock_fw", ["foo"])
  mod_nn = mock_module_with_members(monkeypatch, "mock_fw.nn", ["LogSoftmax", "ReLU"])

  def import_side_effect(name: str) -> types.ModuleType:
    """Execute `ImportError` to support import side effect in discovery inference tests.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `types.ModuleType` from import side effect.
    """
    if name == "mock_fw":
      return mod_root
    if name == "mock_fw.nn":
      return mod_nn
    raise ImportError(f"No mock for {name}")

  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", side_effect=import_side_effect),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("LogSoftmax")
  assert result == "mock_fw.nn.LogSoftmax"


def test_normalized_match(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify case/underscore-normalized matching resolves snake_case operations.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter()
  mod_nn = mock_module_with_members(monkeypatch, "mock_fw.nn", ["log_softmax"])
  mod_root = mock_module_with_members(monkeypatch, "mock_fw", [])

  def side_effect(name: str) -> types.ModuleType:
    """Execute `ImportError` to support side effect in discovery inference tests.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `types.ModuleType` from side effect.
    """
    if name == "mock_fw.nn":
      return mod_nn
    if name == "mock_fw":
      return mod_root
    raise ImportError(name)

  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", side_effect=side_effect),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("LogSoftmax")
  assert result == "mock_fw.nn.log_softmax"


def test_fuzzy_match(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify fuzzy matching resolves close symbol variants when exact match is absent.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter(modules=["mock_fw"])
  mod = mock_module_with_members(monkeypatch, "mock_fw", ["softmax_v2"])
  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", return_value=mod),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("softmax")
  assert result == "mock_fw.softmax_v2"


def test_no_match_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify discover returns None when no candidate matches.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter()
  mod = mock_module_with_members(monkeypatch, "mock_fw", ["nothing_relevant"])
  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", return_value=mod),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("SpecificOp")
  assert result is None


def test_missing_adapter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify SimulatedReflection falls back to root module name when no adapter is registered.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  mod = mock_module_with_members(monkeypatch, "ghost_fw", ["Op"])
  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=None),
    patch("importlib.import_module", return_value=mod) as mock_import,
  ):
    reflector = SimulatedReflection("ghost_fw")
    assert reflector.search_modules == ["ghost_fw"]
    result: str | None = reflector.discover("Op")
    assert result == "ghost_fw.Op"
    mock_import.assert_called_with("ghost_fw")


def test_import_error_handled_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify broken submodule imports are skipped while healthy submodules still resolve.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter(modules=["bad_mod", "good_mod"])
  mod_good = mock_module_with_members(monkeypatch, "good_mod", ["Target"])

  def side_effect(name: str) -> types.ModuleType | None:
    """Execute `ImportError` to support side effect in discovery inference tests.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `types.ModuleType | None` from side effect.
    """
    if name == "bad_mod":
      raise ImportError("Broken")
    if name == "good_mod":
      return mod_good
    return None

  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", side_effect=side_effect),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("Target")
  assert result == "good_mod.Target"


def test_fuzzy_match_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify fuzzy matching skips broken submodule imports gracefully.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = get_mock_adapter(modules=["bad_mod", "good_mod"])
  mod_good = mock_module_with_members(monkeypatch, "good_mod", ["Target_v2"])

  def side_effect(name: str) -> types.ModuleType | None:
    """Execute `ImportError` to support side effect in discovery inference tests.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `types.ModuleType | None` from side effect.
    """
    if name == "bad_mod":
      raise ImportError("Broken")
    if name == "good_mod":
      return mod_good
    return None

  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", side_effect=side_effect),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("Target")
  assert result == "good_mod.Target_v2"


def test_fuzzy_match_no_candidates() -> None:
  """Verify `get_mock_adapter` handles fuzzy match no candidates as expected."""
  adapter = get_mock_adapter(modules=["bad_mod"])

  def side_effect(name: str) -> None:
    """Execute `ImportError` to support side effect in discovery inference tests.

    Args:
        name: Identifier or operation name string.
    """
    raise ImportError("Broken")

  with (
    patch("ml_switcheroo.core.discovery.get_adapter", return_value=adapter),
    patch("importlib.import_module", side_effect=side_effect),
  ):
    reflector = SimulatedReflection("test_fw")
    result: str | None = reflector.discover("Target")
  assert result is None
