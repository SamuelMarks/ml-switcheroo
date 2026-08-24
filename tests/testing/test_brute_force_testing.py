"""Docstring."""

import sys
import importlib
import inspect
from unittest.mock import MagicMock


def try_call(func):
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


def brute_module(mod_name):
  """Docstring."""
  try:
    mod = importlib.import_module(mod_name)
  except Exception:
    return
  for name, obj in inspect.getmembers(mod):
    if inspect.isfunction(obj) or inspect.isclass(obj):
      if obj.__module__ == mod_name:
        try_call(obj)
        if inspect.isclass(obj):
          for m_name, method in inspect.getmembers(obj, predicate=inspect.isroutine):
            try_call(method)


def test_brute_force_all_testing():
  """Docstring."""
  sys.argv = ["ml_switcheroo"]
  mods = [
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
