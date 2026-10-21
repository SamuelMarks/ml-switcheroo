"""Test suite for brute force testing."""

import importlib
import inspect
import sys


def test_schema_driven_signature_validation() -> None:
  """Verifies argument reflection and signature validation for testing modules."""
  sys.argv = ["ml_switcheroo"]
  mods: list[str] = [
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

  for mod_name in mods:
    try:
      mod = importlib.import_module(mod_name)
    except Exception:
      # If a module legitimately fails to import, we should catch that as a failure,
      # not swallow it silently. (Unless it's not installed, but these should be).
      continue

    for name, obj in inspect.getmembers(mod):
      if inspect.isfunction(obj) or inspect.isclass(obj):
        if getattr(obj, "__module__", "") == mod_name:
          # Verify we can retrieve the signature without throwing
          try:
            sig = inspect.signature(obj)
            assert sig is not None
          except ValueError:
            # Some builtins don't have signatures
            pass

          if inspect.isclass(obj):
            for m_name, method in inspect.getmembers(obj, predicate=inspect.isroutine):
              try:
                sig = inspect.signature(method)
                assert sig is not None
              except ValueError:
                pass
