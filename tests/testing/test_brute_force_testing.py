"""Test suite for brute force testing."""

import contextlib
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

  imported_mods: list[str] = []
  signatures_by_symbol: dict[str, tuple[str, ...]] = {}

  for mod_name in mods:
    try:
      mod = importlib.import_module(mod_name)
    except Exception:
      # If a module legitimately fails to import, we should catch that as a failure,
      # not swallow it silently. (Unless it's not installed, but these should be).
      continue
    imported_mods.append(mod_name)

    for name, obj in inspect.getmembers(mod):
      if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == mod_name:
        # Verify we can retrieve the signature without throwing
        with contextlib.suppress(ValueError):
          # Some builtins don't have signatures
          sig = inspect.signature(obj)
          assert sig is not None
          assert isinstance(sig, inspect.Signature)
          signatures_by_symbol[f"{mod_name}.{name}"] = tuple(sig.parameters.keys())

        if inspect.isclass(obj):
          for m_name, method in inspect.getmembers(obj, predicate=inspect.isroutine):
            with contextlib.suppress(ValueError):
              sig = inspect.signature(method)
              assert sig is not None
              assert isinstance(sig, inspect.Signature)
              signatures_by_symbol[f"{mod_name}.{name}.{m_name}"] = tuple(sig.parameters.keys())

  assert len(imported_mods) == len(mods)
  assert signatures_by_symbol["ml_switcheroo.testing.linter.validate_transpilation"] == ("code", "source_framework")
  assert signatures_by_symbol["ml_switcheroo.testing.fuzzer.type_parser.parse_type_annotation"] == ("type_str",)
  assert signatures_by_symbol["ml_switcheroo.testing.fuzzer.core.InputFuzzer.build_strategies"] == (
    "self",
    "params",
    "hints",
    "constraints",
  )
