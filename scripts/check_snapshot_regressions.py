"""Snapshot Regression and Diff Checking Utility.

This script compares upstream framework snapshots or dynamic extracts using
`ml_ecosystem_snapshots.diff` to identify API removals, breaking signature changes,
and type modifications. It generates changelog reports and fails CI when regressions
occur.
"""

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

try:
  from ml_ecosystem_snapshots.diff import diff_snapshots, generate_changelog
  from ml_ecosystem_snapshots.api import extract_snapshot_isolated
except ImportError:  # pragma: no cover
  diff_snapshots = None  # type: ignore[assignment]
  generate_changelog = None  # type: ignore[assignment]
  extract_snapshot_isolated = None  # type: ignore[assignment]


def load_snapshot_file(path: Union[str, Path]) -> Dict[str, Any]:
  """Load snapshot data from a JSON or GZ-compressed file.

  Args:
      path: Path to the snapshot file.

  Returns:
      Dictionary containing loaded snapshot data.

  Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If the file content cannot be parsed as JSON.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"Snapshot file not found: {p}")

  if str(p).endswith(".gz"):
    with gzip.open(p, "rt", encoding="utf-8") as f:
      data = json.load(f)
  else:
    with open(p, "r", encoding="utf-8") as f:
      data = json.load(f)

  if isinstance(data, dict):
    return data
  if isinstance(data, list):
    return {"categories": {"all": data}}
  raise ValueError(f"Invalid snapshot data structure in {p}: expected dict or list, got {type(data).__name__}")


def compare_snapshots(snap1: Any, snap2: Any) -> Any:
  """Diff two snapshot data structures using ml_ecosystem_snapshots.

  Args:
      snap1: Older snapshot data dictionary.
      snap2: Newer snapshot data dictionary.

  Returns:
      A DiffResult object, or None if diff engine is unavailable.
  """
  if diff_snapshots is None:
    return None
  return diff_snapshots(snap1, snap2)


def render_markdown_changelog(diff: Any) -> str:
  """Generate a Markdown formatted changelog report from a DiffResult.

  Args:
      diff: DiffResult instance.

  Returns:
      Formatted markdown changelog string.
  """
  if generate_changelog is None or diff is None:
    return "# Changelog Report\n\nDiff engine not available.\n"
  return str(generate_changelog(diff))


def has_breaking_changes(diff: Any) -> bool:
  """Determine whether a DiffResult contains breaking API changes.

  A change is considered breaking if APIs were removed or breaking signature
  changes were detected.

  Args:
      diff: DiffResult instance.

  Returns:
      True if breaking changes exist, False otherwise.
  """
  if diff is None:
    return False
  removed = getattr(diff, "removed", [])
  breaking_sig = getattr(diff, "breaking_signature_changed", [])
  return bool(removed or breaking_sig)


def run_regression_check(
  older: Optional[str] = None,
  newer: Optional[str] = None,
  framework: Optional[str] = None,
  fail_on_breaking: bool = False,
  output_changelog: Optional[str] = None,
) -> int:
  """Execute snapshot regression check comparing two snapshots or dynamic extraction.

  Args:
      older: Path to older snapshot file.
      newer: Path to newer snapshot file.
      framework: Framework name to dynamically inspect in an isolated subprocess.
      fail_on_breaking: If True, returns non-zero code on breaking changes.
      output_changelog: Path to file where markdown changelog should be saved.

  Returns:
      Exit code: 0 on success/no regressions, 1 on breaking changes or failures.
  """
  if not older:
    print("Error: --older snapshot file is required.", file=sys.stderr)
    return 1

  try:
    snap_old = load_snapshot_file(older)
  except Exception as e:
    print(f"Error loading older snapshot: {e}", file=sys.stderr)
    return 1

  if newer:
    try:
      snap_new = load_snapshot_file(newer)
    except Exception as e:
      print(f"Error loading newer snapshot: {e}", file=sys.stderr)
      return 1
  elif framework:
    if extract_snapshot_isolated is None:
      print("Error: extract_snapshot_isolated not available in environment.", file=sys.stderr)
      return 1
    print(f"Extracting live snapshot for framework '{framework}' in isolated child subprocess...")
    snap_new = extract_snapshot_isolated(framework)
    if not snap_new:
      print(f"Error: failed to extract live snapshot for '{framework}'.", file=sys.stderr)
      return 1
  else:
    print("Error: either --newer snapshot or --framework must be specified.", file=sys.stderr)
    return 1

  diff = compare_snapshots(snap_old, snap_new)
  changelog = render_markdown_changelog(diff)

  if output_changelog:
    out_path = Path(output_changelog)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(changelog)
    print(f"Wrote changelog to {out_path}")

  print(changelog)

  if has_breaking_changes(diff):
    print("⚠️ ALERT: Breaking changes detected in upstream snapshot comparison!", file=sys.stderr)
    if fail_on_breaking:
      return 1

  return 0


def main(argv: Optional[List[str]] = None) -> int:
  """Entry point for the snapshot regression check CLI.

  Args:
      argv: Optional list of command-line argument strings.

  Returns:
      Integer process exit status code.
  """
  parser = argparse.ArgumentParser(
    description="Compare ML framework snapshots and verify upstream API regressions.",
  )
  parser.add_argument(
    "--older",
    required=True,
    help="Path to older/baseline snapshot file (.json or .json.gz).",
  )
  parser.add_argument(
    "--newer",
    default=None,
    help="Path to newer snapshot file (.json or .json.gz).",
  )
  parser.add_argument(
    "--framework",
    default=None,
    help="Target framework name to dynamically extract in isolated subprocess if --newer is omitted.",
  )
  parser.add_argument(
    "--fail-on-breaking",
    action="store_true",
    help="Exit with non-zero status code if breaking API changes or removals are detected.",
  )
  parser.add_argument(
    "--output-changelog",
    default=None,
    help="Optional path to output markdown changelog report.",
  )

  args = parser.parse_args(argv)
  return run_regression_check(
    older=args.older,
    newer=args.newer,
    framework=args.framework,
    fail_on_breaking=args.fail_on_breaking,
    output_changelog=args.output_changelog,
  )


if __name__ == "__main__":
  sys.exit(main())
