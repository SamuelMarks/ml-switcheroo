"""
Tests for Robust Varargs Support (Feature 07).

Verifies that:
1. ApiInspector correctly flags functions with `*args` as `has_varargs=True`.
2. Scaffolder waives arity penalties when `has_varargs` is True.
"""

import pytest
import json
from unittest.mock import MagicMock
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager

# --- Mocks for Inspector Test ---


class MockParameter:
  def __init__(self, name, kind_str=None):
    self.name = name
    self.kind = MagicMock()
    self.kind.__str__.return_value = kind_str if kind_str else ""


def test_inspector_detects_varargs():
  """
  Scenario: Analyzes a function with and without *args using Mock Griffe behavior.
  """
  inspector = ApiInspector()

  # 1. Standard Function: f(x, y)
  func_std = MagicMock()
  func_std.name = "std_func"
  func_std.parameters = [MockParameter("x", "POSITIONAL_OR_KEYWORD"), MockParameter("y", "POSITIONAL_OR_KEYWORD")]
  func_std.docstring = None

  sig_std = inspector._extract_signature(func_std, "function")
  assert sig_std["params"] == ["x", "y"]
  assert sig_std["has_varargs"] is False

  # 2. Varargs Function: f(*args)
  # Griffe represents *args using ParameterKind.VAR_POSITIONAL
  func_var = MagicMock()
  func_var.name = "var_func"
  func_var.parameters = [MockParameter("args", "VAR_POSITIONAL")]
  func_var.docstring = None

  sig_var = inspector._extract_signature(func_var, "function")
  assert "args" in sig_var["params"]
  assert sig_var["has_varargs"] is True

  # 3. Asterisk Syntax fallback: f(*items)
  # If kind string check fails but name starts with *, we should catch it
  func_legacy = MagicMock()
  func_legacy.name = "legacy_func"
  func_legacy.parameters = [MockParameter("*items", "UNKNOWN_KIND")]
  func_legacy.docstring = None

  sig_legacy = inspector._extract_signature(func_legacy, "function")
  assert sig_legacy["has_varargs"] is True
  assert "items" in sig_legacy["params"]  # Checked strict cleanup


# --- Scaffolder Tests ---


class MockVarargsInspector(ApiInspector):
  """Returns catalogs with constrained arity scenarios."""

  def inspect(self, fw_name: str) -> dict:
    if fw_name == "source_fw":
      return {
        "source.add": {
          "name": "add",
          "params": ["x", "y"],  # Arity 2
          "has_varargs": False,
        }
      }

    if fw_name == "target_fw":
      return {
        # Case A: Mismatch Arity (1) - Fails without varargs
        # We name it 'target.strict.add' so leaf name is 'add' (matching source)
        "target.strict.add": {
          "name": "add",
          "params": ["a"],  # Arity 1
          "has_varargs": False,
        },
        # Case B: Mismatch Arity (1) - Passes with varargs
        # This mimics numpy.add(*args) wrappers
        "target.poly.add": {
          "name": "add",
          "params": ["args"],  # Arity 1 (the args tuple)
          "has_varargs": True,
        },
      }
    return {}


@pytest.fixture
def clean_semantics():
  """Returns a manager with no pre-loaded data."""
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._key_origins = {}
  return mgr


def test_scaffolder_skips_penalty_for_varargs(tmp_path, clean_semantics):
  """
  Scenario:
      Source: add(x, y) [Arity 2]
      Target 1: poly_add(*args) [Arity 1, has_varargs=True]
      Target 2: strict_add(a)   [Arity 1, has_varargs=False]

  Expectation:
      - strict_add is penalized (Arity mismatch 2 vs 1) -> Score < Threshold
      - poly_add is NOT penalized (Varargs waives mismatch) -> Score > Threshold
      - poly_add is chosen.
  """
  scaffolder = Scaffolder(semantics=clean_semantics, similarity_threshold=0.8, arity_penalty=0.5)
  scaffolder.inspector = MockVarargsInspector()

  scaffolder.scaffold(["source_fw", "target_fw"], tmp_path)

  out_file = tmp_path / "k_array_api.json"
  with open(out_file, "rt", encoding="utf-8") as f:
    data = json.load(f)

  # Because semantics are clean, 'add' defaults to Array API tier
  assert "add" in data
  variants = data["add"]["variants"]

  # Ensure we matched
  assert "target_fw" in variants

  # Ensure we matched the poly_add (varargs) version, not the strict one
  matched_api = variants["target_fw"]["api"]
  assert matched_api == "target.poly.add"
