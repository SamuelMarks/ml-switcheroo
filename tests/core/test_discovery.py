"""Unit tests for `SimulatedReflection` and `is_inspection_object` (`ml_switcheroo.core.discovery`)."""

import importlib
import math

import pytest

import ml_switcheroo.core.discovery as discovery_mod
from ml_switcheroo.core.discovery import SimulatedReflection
from ml_switcheroo.discovery.inspector import is_inspection_object


class _AdapterWithModules:
  """Typed adapter stub carrying `search_modules`."""

  def __init__(self, search_modules: list[str]) -> None:
    """Initialize `search_modules` list.

    Args:
        search_modules: Module names searched during reflection.
    """
    self.search_modules = search_modules


class _AdapterWithoutModules:
  """Typed adapter stub omitting `search_modules`."""


def test_simulated_reflection_initialization_and_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SimulatedReflection` module fallback resolution and `_normalize` canonicalization.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr("ml_switcheroo.core.discovery.get_adapter", lambda _fw: _AdapterWithModules(["mod1", "mod2"]))
  assert SimulatedReflection("custom_fw").search_modules == ["mod1", "mod2"]

  monkeypatch.setattr("ml_switcheroo.core.discovery.get_adapter", lambda _fw: _AdapterWithoutModules())
  assert SimulatedReflection("custom_fw").search_modules == ["custom_fw"]

  monkeypatch.setattr("ml_switcheroo.core.discovery.get_adapter", lambda _fw: None)
  ref = SimulatedReflection("custom_fw")
  assert ref.framework == "custom_fw"
  assert ref.search_modules == ["custom_fw"]
  assert ref._normalize("Log_Softmax") == "logsoftmax"
  assert ref._normalize("abs_") == "abs"


def test_simulated_reflection_exact_fuzzy_and_import_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify exact normalized matching, fuzzy fallback, private skipping, and `ImportError` handling.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """

  class _FakeModNamespace:
    """Simulated module namespace exposing `_private`, `log_softmax`, and `absolute`."""

    @staticmethod
    def _private() -> None:
      """Private attribute skipped by `SimulatedReflection`."""

    @staticmethod
    def log_softmax() -> None:
      """Exact-normalized candidate attribute."""

    @staticmethod
    def absolute() -> None:
      """Fuzzy-matched candidate attribute."""

  fake_mod = _FakeModNamespace()

  def _fake_import(name: str) -> _FakeModNamespace:
    """Resolve `'fake_mod'` in memory and raise `ImportError` for `'broken_mod'`.

    Args:
        name: Module path string.

    Returns:
        In-memory `_FakeModNamespace` for `'fake_mod'`.

    Raises:
        ImportError: When `name != 'fake_mod'`.
    """
    if name == "fake_mod":
      return fake_mod
    raise ImportError(name)

  monkeypatch.setattr(
    discovery_mod,
    "get_adapter",
    lambda _fw: _AdapterWithModules(["broken_mod", "fake_mod"]),
  )
  monkeypatch.setattr(importlib, "import_module", _fake_import)

  ref = SimulatedReflection("custom_fw")
  assert ref.discover("Log_Softmax") == "fake_mod.log_softmax"
  assert ref.discover("absolut") == "fake_mod.absolute"
  assert ref.discover("completely_unrelated_op_name") is None

  # All modules raising ImportError returns None cleanly
  monkeypatch.setattr(
    discovery_mod,
    "get_adapter",
    lambda _fw: _AdapterWithModules(["broken_mod"]),
  )
  assert SimulatedReflection("custom_fw").discover("op") is None


def test_is_inspection_object_classifications() -> None:
  """Verify `is_inspection_object` recognizes modules, classes, functions, builtins, and properties."""

  def _prop_getter(_self: _AdapterWithoutModules) -> int:
    """Property getter stub.

    Args:
        _self: Receiver instance.

    Returns:
        Constant `1`.
    """
    return 1

  assert is_inspection_object(math) is True
  assert is_inspection_object(_AdapterWithoutModules) is True
  assert is_inspection_object(test_is_inspection_object_classifications) is True
  assert is_inspection_object(math.sin) is True
  assert is_inspection_object(property(_prop_getter)) is True
  assert is_inspection_object(42) is False
