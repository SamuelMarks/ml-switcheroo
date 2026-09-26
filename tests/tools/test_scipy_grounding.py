"""Test suite for SciPy Fallback Grounding and Validation Utilities."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from ml_switcheroo.tools.injector_fw.scipy_grounding import (
  audit_injected_scipy_fallbacks,
  extract_scipy_endpoints_from_macro,
  load_scipy_snapshot,
  validate_scipy_api,
  validate_scipy_macro,
)


def test_load_scipy_snapshot_custom(tmp_path: Path) -> None:
  """Tests loading and indexing a custom SciPy snapshot JSON file."""
  snap_data = {
    "categories": {
      "special": [
        {"api_path": "scipy.special.i0", "name": "i0"},
        {"api_path": "scipy.special.erf", "name": "erf"},
        "skip_non_dict",
        {"name": ""},
      ],
      "invalid_cat": "not_a_list",
    }
  }
  snap_file = tmp_path / "scipy_v1.13.1.json"
  snap_file.write_text(json.dumps(snap_data))

  index = load_scipy_snapshot(snap_file)
  assert "scipy.special.i0" in index
  assert "i0" in index
  assert "scipy.special.erf" in index

  # Non-dict categories
  snap_file_bad = tmp_path / "scipy_bad.json"
  snap_file_bad.write_text(json.dumps({"categories": "not_a_dict"}))
  assert load_scipy_snapshot(snap_file_bad) == {}


def test_scipy_grounding_defaults(tmp_path: Path) -> None:
  """Tests calling functions with default snapshot (snapshot=None)."""
  snap_file = tmp_path / "scipy_v1.13.1.json"
  snap_file.write_text(
    json.dumps(
      {
        "categories": {
          "special": [
            {"api_path": "scipy.special.i0", "name": "i0"},
          ]
        }
      }
    )
  )

  with patch("ml_switcheroo.tools.injector_fw.scipy_grounding.resolve_snapshots_dir", return_value=tmp_path):
    # load_scipy_snapshot() with default path
    idx = load_scipy_snapshot()
    assert "i0" in idx

    # validate_scipy_api with default snapshot
    assert validate_scipy_api("scipy.special.i0")
    assert validate_scipy_api("i0")

    # validate_scipy_macro with default snapshot
    assert validate_scipy_macro("scipy.special.i0({x})")

    # audit_injected_scipy_fallbacks with default snapshot
    errs = audit_injected_scipy_fallbacks({"jax": {"api": "scipy.special.i0"}})
    assert errs == []


def test_extract_scipy_endpoints_attribute_non_name() -> None:
  """Tests extract_scipy_endpoints_from_macro with attribute chain ending on non-Name."""
  macro = "(123).scipy.special.i0"
  endpoints = extract_scipy_endpoints_from_macro(macro)
  assert endpoints == set()


def test_load_scipy_snapshot_errors(tmp_path: Path) -> None:
  """Tests error handling for missing SciPy snapshot files."""
  missing_file = tmp_path / "non_existent.json"
  with pytest.raises(FileNotFoundError, match="does not exist"):
    load_scipy_snapshot(missing_file)

  empty_dir = tmp_path / "empty_snaps"
  empty_dir.mkdir()
  with patch("ml_switcheroo.tools.injector_fw.scipy_grounding.resolve_snapshots_dir", return_value=empty_dir):
    with pytest.raises(FileNotFoundError, match="No SciPy snapshot found"):
      load_scipy_snapshot()


def test_extract_scipy_endpoints_from_macro() -> None:
  """Tests extraction of SciPy endpoints from templates and handling of syntax errors."""
  macro1 = "scipy.special.i0({x}) + jax.scipy.special.erf({y.shape[0]})"
  endpoints1 = extract_scipy_endpoints_from_macro(macro1)
  assert "scipy.special.i0" in endpoints1
  assert "jax.scipy.special.erf" in endpoints1

  # Non-scipy call
  macro2 = "torch.abs({x}) + math.sqrt(2.0)"
  endpoints2 = extract_scipy_endpoints_from_macro(macro2)
  assert len(endpoints2) == 0

  # Syntax error in macro
  macro_bad = "def invalid (("
  assert extract_scipy_endpoints_from_macro(macro_bad) == set()


def test_validate_scipy_api_and_macro() -> None:
  """Tests validation of APIs and macro templates against a mock snapshot."""
  mock_snapshot = {
    "scipy.special.i0": {"name": "i0"},
    "i0": {"name": "i0"},
    "scipy.special.erf": {"name": "erf"},
    "erf": {"name": "erf"},
    "scipy.linalg.cholesky": {"name": "cholesky"},
    "cholesky": {"name": "cholesky"},
  }

  # Valid calls
  assert validate_scipy_api("scipy.special.i0", mock_snapshot)
  assert validate_scipy_api("jax.scipy.special.erf", mock_snapshot)
  assert validate_scipy_api("prefix.cholesky", {"cholesky": {"name": "cholesky"}})
  assert validate_scipy_api("cholesky", mock_snapshot)

  # Invalid calls
  with pytest.raises(ValueError, match="Ungrounded SciPy endpoint"):
    validate_scipy_api("scipy.special.unsupported_op", mock_snapshot)

  # Valid macro
  macro_valid = "scipy.special.i0({x}) + scipy.special.erf({x})"
  assert validate_scipy_macro(macro_valid, mock_snapshot)

  # Invalid macro
  macro_invalid = "scipy.special.fake_op({x})"
  with pytest.raises(ValueError, match="Ungrounded SciPy endpoint"):
    validate_scipy_macro(macro_invalid, mock_snapshot)


def test_audit_injected_scipy_fallbacks() -> None:
  """Tests auditing a dictionary of framework variants for ungrounded SciPy usages."""
  mock_snapshot = {
    "scipy.special.i0": {"name": "i0"},
    "scipy.special.erf": {"name": "erf"},
  }

  variants_valid = {
    "jax": {"api": "jax.scipy.special.erf"},
    "torch": {"macro_template": "scipy.special.i0({x})"},
    "ignored": "not_a_dict",
    "numpy": {"api": "numpy.abs"},
  }
  errors = audit_injected_scipy_fallbacks(variants_valid, mock_snapshot)
  assert len(errors) == 0

  variants_invalid = {
    "flax": {"api": "scipy.special.non_existent"},
    "mlx": {"macro_template": "scipy.special.another_fake({x})"},
  }
  errors_bad = audit_injected_scipy_fallbacks(variants_invalid, mock_snapshot)
  assert len(errors_bad) == 2
  assert any("[flax]" in err for err in errors_bad)
  assert any("[mlx]" in err for err in errors_bad)

  # Failure loading snapshot
  with patch(
    "ml_switcheroo.tools.injector_fw.scipy_grounding.load_scipy_snapshot", side_effect=RuntimeError("disk error")
  ):
    errs = audit_injected_scipy_fallbacks({})
    assert len(errs) == 1
    assert "Failed to load SciPy snapshot" in errs[0]


def test_bessel_and_error_functions_golden() -> None:
  """Verifies that all standard Bessel and error functions exist in real snapshot if present."""
  from ml_switcheroo.semantics.paths import resolve_snapshots_dir

  snap_dir = resolve_snapshots_dir()
  candidates = sorted(snap_dir.glob("scipy_v*.json"))
  if not candidates:
    return

  index = load_scipy_snapshot(candidates[0])
  # Bessel routines
  for bessel_fn in ("i0", "i1", "j0", "j1", "k0", "k1", "y0", "y1"):
    assert bessel_fn in index or f"scipy.special.{bessel_fn}" in index

  # Error and special math routines
  for special_fn in ("erf", "erfc", "gammaln", "digamma", "polygamma", "betaln", "logit", "logsumexp"):
    assert special_fn in index or f"scipy.special.{special_fn}" in index
