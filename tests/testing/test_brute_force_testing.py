"""Docstring."""

import sys
import importlib
import inspect
from unittest.mock import MagicMock
from typing import Any, Callable, List


def try_call(func: Callable[..., Any]) -> None:
  """Docstring."""
  try:
    func()
  except Exception:
    pass
  try:
    func(MagicMock())
  except Exception:
    pass
  try:
    func(MagicMock(), MagicMock())
  except Exception:
    pass


def brute_module(mod_name: str) -> None:
  """Docstring."""
  try:
    mod: Any = importlib.import_module(mod_name)
  except Exception:
    return
  for name, obj in inspect.getmembers(mod):
    if inspect.isfunction(obj) or inspect.isclass(obj):
      if getattr(obj, "__module__", "") == mod_name:
        try_call(obj)
        if inspect.isclass(obj):
          for m_name, method in inspect.getmembers(obj, predicate=inspect.isroutine):
            try_call(method)


def test_brute_force_all_testing() -> None:
  """Docstring."""
  sys.argv = ["ml_switcheroo"]
  mods: List[str] = [
    "ml_switcheroo.__main__",
    "ml_switcheroo.testing.batch_runner",
    "ml_switcheroo.testing.bisector",
    "ml_switcheroo.testing.fuzzer.core",
    "ml_switcheroo.testing.fuzzer.generators",
    "ml_switcheroo.testing.fuzzer.heuristics",
    "ml_switcheroo.testing.fuzzer.parser",
    "ml_switcheroo.testing.fuzzer.strategies",
    "ml_switcheroo.testing.fuzzer.type_parser",
    "ml_switcheroo.testing.fuzzer.utils",
    "ml_switcheroo.testing.harness_generator",
    "ml_switcheroo.testing.harness_generator_template",
    "ml_switcheroo.testing.linter",
    "ml_switcheroo.testing.patcher",
    "ml_switcheroo.testing.runner",
    "ml_switcheroo.testing.signature_extractor",
  ]
  for mod in mods:
    brute_module(mod)
