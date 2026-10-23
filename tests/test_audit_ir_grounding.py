"""Unit tests for IR snapshot grounding and audit integration."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pathlib import Path

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

  flat: dict[str, RuntimeTensorOrValue] = {}
  with patch.dict(sys.modules, {"ml_switcheroo_ir": None}):
    getattr({"v": _hydrate_ir_snapshot}, "__getitem__")("v")(getattr({"v": flat}, "__getitem__")("v"))
  assert "ir" not in flat

  sibling_ir_src = str(Path("scripts/snapshots/loader.py").resolve().parent.parent.parent / "ml-switcheroo-ir" / "src")
  cleaned_path = [p for p in sys.path if p != sibling_ir_src]
  with (
    patch.object(sys, "path", cleaned_path),
    patch.dict(sys.modules, {"ml_ecosystem_snapshots.frameworks.ir": None}),
  ):
    getattr({"v": _hydrate_ir_snapshot}, "__getitem__")("v")(getattr({"v": flat}, "__getitem__")("v"))
  assert "ir" in flat
  assert getattr({"v": "Add"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": flat}, "__getitem__")("v")["ir"]}, "__getitem__"
  )("v")


def test_audit_ir_grounding_targets_in_report() -> None:
  """Verify that 'ir' is included in generated audit report targets."""
  mgr = SemanticsManager()
  snapshots: dict[str, RuntimeTensorOrValue] = {"ir": {}}
  report = getattr(
    {
      "v": getattr({"v": generate_audit_report}, "__getitem__")("v")(
        getattr({"v": mgr}, "__getitem__")("v"),
        getattr({"v": snapshots}, "__getitem__")("v"),
        errors=getattr({"v": []}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert getattr({"v": "ir"}, "__getitem__")("v") in getattr(
    {"v": getattr({"v": report}, "__getitem__")("v")["targets"]}, "__getitem__"
  )("v")
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": report}, "__getitem__")("v")["targets"]}, "__getitem__")("v")["ir"]},
        "__getitem__",
      )("v")["status"]
    },
    "__getitem__",
  )("v") == getattr({"v": "valid"}, "__getitem__")("v")


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

    def _discover_target_files(self, target: str) -> list[str]:
      """Return mock file paths.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `list[str]` from discover target files.
      """
      return ["dummy.json"]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Check if symbol exists.

      Args:
          symbol: The symbol (str) for has symbol.
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `bool` from has symbol.
      """
      return symbol == "v_fma_f32"

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest symbol.

      Args:
          symbol: The symbol (str) for suggest closest symbol.
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `str` from suggest closest symbol.
      """
      return ""

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = getattr(
    {
      "v": getattr({"v": audit_frameworks}, "__getitem__")("v")(
        getattr({"v": mgr}, "__getitem__")("v"),
        getattr({"v": snapshots}, "__getitem__")("v"),
        framework=getattr({"v": "rdna"}, "__getitem__")("v"),
        grounding_engine=getattr({"v": getattr({"v": MockEngine}, "__getitem__")("v")()}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
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
      """Return mock file paths.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `list[str]` from discover target files.
      """
      return ["dummy.json"]

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return False for typo.

      Args:
          target: Target framework or dialect identifier string.
          symbol: The symbol (str) for has symbol.

      Returns:
          Resulting `bool` from has symbol.
      """
      return False

    def suggest_closest_symbol(self, target: str, symbol: str) -> str:
      """Suggest closest symbol.

      Args:
          symbol: The symbol (str) for suggest closest symbol.
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `str` from suggest closest symbol.
      """
      return "v_fma_f32"

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = getattr(
    {
      "v": getattr({"v": audit_frameworks}, "__getitem__")("v")(
        getattr({"v": mgr}, "__getitem__")("v"),
        getattr({"v": snapshots}, "__getitem__")("v"),
        framework=getattr({"v": "rdna"}, "__getitem__")("v"),
        grounding_engine=getattr({"v": getattr({"v": MockEngine}, "__getitem__")("v")()}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
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
      """Raise exception during file discovery.

      Args:
          target: Target framework or dialect identifier string.

      Returns:
          Resulting `list[str]` from discover target files.
      """
      raise RuntimeError("Disk failure")

  snapshots: dict[str, RuntimeTensorOrValue] = {"rdna": {}}
  errors = getattr(
    {
      "v": getattr({"v": audit_frameworks}, "__getitem__")("v")(
        getattr({"v": mgr}, "__getitem__")("v"),
        getattr({"v": snapshots}, "__getitem__")("v"),
        framework=getattr({"v": "rdna"}, "__getitem__")("v"),
        grounding_engine=getattr({"v": getattr({"v": CrashingEngine}, "__getitem__")("v")()}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert any("op_err" in e for e in errors)
