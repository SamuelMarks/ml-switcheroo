"""Unit tests for IR snapshot grounding and audit integration."""

from pathlib import Path
from typing import Any, Dict

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
    Path("src/ml_switcheroo/semantics"),
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)
  errors = audit_frameworks(mgr, snapshots, framework="ir")
  assert isinstance(errors, list)


def test_audit_ir_grounding_targets_in_report() -> None:
  """Verify that 'ir' is included in generated audit report targets."""
  mgr = SemanticsManager()
  snapshots: Dict[str, Dict[str, Any]] = {"ir": {}}
  report = generate_audit_report(mgr, snapshots, errors=[])
  assert "ir" in report["targets"]
  assert report["targets"]["ir"]["status"] == "valid"


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
  snapshots: Dict[str, Dict[str, Any]] = {
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

    def _discover_target_files(self, target: str) -> list[str]:
      """Return mock file paths."""
      return ["dummy.json"]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Check if symbol exists."""
      return symbol == "v_fma_f32"

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest symbol."""
      return ""

  snapshots: Dict[str, Dict[str, Any]] = {"rdna": {}}
  errors = audit_frameworks(mgr, snapshots, framework="rdna", grounding_engine=MockEngine())
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

    def _discover_target_files(self, target: str) -> list[str]:
      """Return mock file paths."""
      return ["dummy.json"]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return False for typo."""
      return False

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest closest symbol."""
      return "v_fma_f32"

  snapshots: Dict[str, Dict[str, Any]] = {"rdna": {}}
  errors = audit_frameworks(mgr, snapshots, framework="rdna", grounding_engine=MockEngine())
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

    def _discover_target_files(self, target: str) -> list[str]:
      """Raise exception during file discovery."""
      raise RuntimeError("Disk failure")

  snapshots: Dict[str, Dict[str, Any]] = {"rdna": {}}
  errors = audit_frameworks(mgr, snapshots, framework="rdna", grounding_engine=CrashingEngine())
  assert any("op_err" in e for e in errors)
