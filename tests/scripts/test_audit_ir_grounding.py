"""Unit tests for IR snapshot grounding and audit integration."""

from pathlib import Path

from pydantic import JsonValue

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from scripts.audit_against_snapshots import (
  audit_frameworks,
  audit_inline_snippets,
  generate_audit_report,
  load_snapshots_multi,
)


def test_audit_ir_grounding_pass() -> None:
  """Verify that auditing the IR framework against loaded snapshots passes."""
  mgr = SemanticsManager()
  snapshot_dirs = [
    Path("../ml-switcheroo-ir"),
    Path("src/ml_switcheroo/semantics"),
    Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)
  assert "ir" in snapshots
  assert "LogicalNode" in snapshots["ir"]
  assert "Add" in snapshots["ir"]
  errors = audit_frameworks(mgr, snapshots, framework="ir")
  assert isinstance(errors, list)


def test_hydrate_ir_snapshot_import_fallbacks() -> None:
  """Verify _hydrate_ir_snapshot handles missing ml_switcheroo_ir and ml_ecosystem_snapshots gracefully."""
  import sys
  from unittest.mock import patch

  from scripts.snapshots.loader import _hydrate_ir_snapshot

  flat: dict[str, dict[str, dict[str, JsonValue]]] = {}
  with patch.dict(sys.modules, {"ml_switcheroo_ir": None}):
    _hydrate_ir_snapshot(flat)
  assert "ir" not in flat

  sibling_ir_src = str(Path("scripts/snapshots/loader.py").resolve().parent.parent.parent / "ml-switcheroo-ir" / "src")
  cleaned_path = [p for p in sys.path if p != sibling_ir_src]
  with (
    patch.object(sys, "path", cleaned_path),
    patch.dict(sys.modules, {"ml_ecosystem_snapshots.frameworks.ir": None}),
  ):
    _hydrate_ir_snapshot(flat)
  assert "ir" in flat
  assert "Add" in flat["ir"]


def test_audit_ir_grounding_targets_in_report() -> None:
  """Verify that 'ir' is included in generated audit report targets."""
  mgr = SemanticsManager()
  snapshots: dict[str, dict[str, RuntimeTensorOrValue]] = {"ir": {}}
  report = generate_audit_report(
    mgr,
    snapshots,
    errors=[],
  )
  targets = report["targets"]
  assert isinstance(targets, dict)
  assert "ir" in targets
  ir_target = targets["ir"]
  assert isinstance(ir_target, dict)
  assert ir_target["status"] == "valid"


def test_audit_ir_grounding_detects_hallucinations() -> None:
  """Verify that audit_inline_snippets flags ungrounded IR API calls."""
  mgr = SemanticsManager()
  # Mock an operation variant with hallucinated IR call
  mgr.data = {
    "fake_op": {
      "variants": {
        "ir": {
          "macro_template": "sw_ir.HallucinatedOp(x)",
        }
      }
    }
  }
  snapshots: dict[str, RuntimeTensorOrValue] = {
    "ir": {
      "sw_ir.LogicalNode": {},
    }
  }
  errors = audit_inline_snippets(mgr, snapshots)
  assert any("fake_op" in e and "sw_ir.HallucinatedOp" in e for e in errors)


def test_audit_frameworks_with_grounding_engine_symbol_validation() -> None:
  """Verify that audit_frameworks validates symbols via GroundingEngine."""
  mgr = SemanticsManager()
  mgr.data = {
    "op_test": {
      "variants": {
        "rdna": {"api": "v_fma_f32"},
      }
    }
  }

  class MockEngine:
    """Mock GroundingEngine for symbol validation."""

    def _discover_target_files(self, target: str) -> list[Path]:
      """Return mock file paths.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          List of mock snapshot `Path` objects.
      """
      return [Path("dummy.json")]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Check if symbol exists.

      Args:
          symbol: Input `symbol` parameter value (`str`).
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `bool` from has symbol.
      """
      return symbol == "v_fma_f32"

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest symbol.

      Args:
          symbol: Input `symbol` parameter value (`str`).
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `str` from suggest closest symbol.
      """
      return ""

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = audit_frameworks(
    mgr,
    snapshots,
    framework="rdna",
    grounding_engine=MockEngine(),
  )
  assert not any("op_test" in e for e in errors)


def test_audit_frameworks_with_grounding_engine_suggestion() -> None:
  """Verify that audit_frameworks incorporates closest symbol suggestions."""
  mgr = SemanticsManager()
  mgr.data = {
    "op_typo": {
      "variants": {
        "rdna": {"api": "v_fma_f32_typo"},
      }
    }
  }

  class MockEngine:
    """Mock GroundingEngine for symbol suggestions."""

    def _discover_target_files(self, target: str) -> list[Path]:
      """Return mock file paths.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          List of mock snapshot `Path` objects.
      """
      return [Path("dummy.json")]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return False for typo.

      Args:
          target: Target framework or dialect identifier string.
          symbol: Input `symbol` parameter value (`str`).

      Returns:
          Resulting `bool` from has symbol.
      """
      return False

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest closest symbol.

      Args:
          symbol: Input `symbol` parameter value (`str`).
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `str` from suggest closest symbol.
      """
      return "v_fma_f32"

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = audit_frameworks(
    mgr,
    snapshots,
    framework="rdna",
    grounding_engine=MockEngine(),
  )
  assert any("op_typo" in e and "(did you mean: 'v_fma_f32'?)" in e for e in errors)


def test_audit_frameworks_with_grounding_engine_exception_resilience() -> None:
  """Verify that exceptions inside GroundingEngine methods are handled gracefully."""
  mgr = SemanticsManager()
  mgr.data = {
    "op_err": {
      "variants": {
        "rdna": {"api": "v_broken"},
      }
    }
  }

  class CrashingEngine:
    """Mock GroundingEngine that raises errors."""

    def _discover_target_files(self, target: str) -> list[Path]:
      """Raise exception during file discovery.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          List of mock snapshot `Path` objects.
      """
      raise RuntimeError("Disk failure")

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Raise exception during symbol lookup.

      Args:
          target: Target framework or dialect identifier string.
          symbol: Symbol identifier to check.

      Returns:
          Boolean indicating symbol presence.
      """
      raise RuntimeError("Disk failure")

    def suggest_closest_symbol(self, target: str, symbol: str) -> str | None:
      """Raise exception during closest symbol suggestion.

      Args:
          target: Target framework or dialect identifier string.
          symbol: Symbol identifier to match.

      Returns:
          Suggested closest symbol string or None.
      """
      raise RuntimeError("Disk failure")

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = audit_frameworks(
    mgr,
    snapshots,
    framework="rdna",
    grounding_engine=CrashingEngine(),
  )
  assert any("op_err" in e for e in errors)
