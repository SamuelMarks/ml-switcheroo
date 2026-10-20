"""Report generation and metrics formatting."""

import typing

from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager


def generate_audit_report(
  manager: SemanticsManager,
  snapshots: typing.Mapping[str, typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]],
  errors: typing.List[str],
  checksums: typing.Optional[typing.Dict[str, str]] = None,
) -> typing.Dict[str, JsonValue]:
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
  target_metrics: typing.Dict[str, JsonValue] = {}

  for target in targets:
    mapped_count = 0
    for _op_name, op_details in manager.data.items():
      variants = op_details.get("variants")
      if isinstance(variants, dict) and target in variants:
        mapped_count += 1

    target_errors = [e for e in errors if f"[{target}]" in e]
    target_metrics[target] = {
      "mapped_operations": mapped_count,
      "snapshot_symbols": len(snapshots.get(target, {})),
      "errors": len(target_errors),
      "status": "valid" if len(target_errors) == 0 else "mismatched",
    }

  loaded_snaps: typing.List[JsonValue] = [str(k) for k in snapshots.keys()]
  checksum_json: typing.Dict[str, JsonValue] = {str(k): str(v) for k, v in (checksums or {}).items()}

  return {
    "total_operations": len(manager.data),
    "loaded_snapshots": loaded_snaps,
    "snapshot_checksums": checksum_json,
    "total_errors": len(errors),
    "status": "pass" if len(errors) == 0 else "fail",
    "targets": target_metrics,
  }
