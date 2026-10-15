"""Report generation and metrics formatting."""

from typing import Any, Dict, List, Optional

from ml_switcheroo.semantics.manager import SemanticsManager


def generate_audit_report(
  manager: SemanticsManager,
  snapshots: Dict[str, Dict[str, Any]],
  errors: List[str],
  checksums: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
  """Generates a structured audit report dictionary across all supported targets.

  Args:
      manager: The active SemanticsManager instance.
      snapshots: Flattened snapshot lookup dictionary.
      errors: List of detected error strings.
      checksums: Optional dictionary of snapshot file SHA-256 checksums.

  Returns:
      A dictionary detailing coverage, mapped operation counts, and error metrics.
  """
  targets = ["torch", "jax", "mlx", "keras", "nvidia_sass", "rdna", "ir", "array_api", "scipy", "safetensors"]
  target_metrics: Dict[str, Dict[str, Any]] = {}

  for target in targets:
    mapped_count = 0
    for _op_name, op_details in manager.data.items():
      variants = op_details.get("variants", {})
      if target in variants:
        mapped_count += 1

    target_errors = [e for e in errors if f"[{target}]" in e]
    target_metrics[target] = {
      "mapped_operations": mapped_count,
      "snapshot_symbols": len(snapshots.get(target, {})),
      "errors": len(target_errors),
      "status": "valid" if len(target_errors) == 0 else "mismatched",
    }

  return {
    "total_operations": len(manager.data),
    "loaded_snapshots": list(snapshots.keys()),
    "snapshot_checksums": checksums or {},
    "total_errors": len(errors),
    "status": "pass" if len(errors) == 0 else "fail",
    "targets": target_metrics,
  }
