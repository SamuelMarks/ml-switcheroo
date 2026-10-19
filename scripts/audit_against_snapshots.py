"""Audits framework mappings against actual extracted snapshots.

Validates that APIs and parameters mapped in ODL/JSON exist in the real snapshots.
"""

import argparse
import json
import sys
from pathlib import Path

# Load local components
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader
from scripts.snapshots.ast_auditor import (
  NVIDIA_SASS_KNOWN_INSTRUCTIONS,
  RDNA_KNOWN_INSTRUCTIONS,
  audit_frameworks,
  audit_inline_snippets,
  audit_python_ast,
  extract_api_calls,
)
from scripts.snapshots.loader import (
  GroundingEngine,
  _flatten_single_framework,
  _resolve_grounding_engine,
  compute_snapshot_checksums,
  load_snapshots,
  load_snapshots_multi,
)
from scripts.snapshots.reporter import generate_audit_report

__all__ = [
  "GroundingEngine",
  "NVIDIA_SASS_KNOWN_INSTRUCTIONS",
  "RDNA_KNOWN_INSTRUCTIONS",
  "_flatten_single_framework",
  "_resolve_grounding_engine",
  "audit_frameworks",
  "audit_inline_snippets",
  "audit_python_ast",
  "compute_snapshot_checksums",
  "extract_api_calls",
  "generate_audit_report",
  "load_snapshots",
  "load_snapshots_multi",
  "main",
]


def main() -> int:
  """Main execution function.

  Returns:
      Exit code (0 for success, 1 for failures).
  """
  parser = argparse.ArgumentParser(description="Audit against snapshots")
  parser.add_argument("--strict", action="store_true", help="Fail if any mismatches found")
  parser.add_argument(
    "--framework", type=str, default=None, help="Audit only a specific framework (e.g. rdna, nvidia_sass)"
  )
  parser.add_argument(
    "--report", "--output", dest="report_path", type=str, default=None, help="Save structured audit report to JSON"
  )
  parser.add_argument(
    "--snapshots-dir",
    action="append",
    default=[],
    help="Explicit snapshot directory to audit against (can be specified multiple times)",
  )
  args = parser.parse_args()

  mgr = SemanticsManager()
  KnowledgeBaseLoader(mgr).load_knowledge_graph()
  RegistryLoader(mgr).hydrate()

  snapshot_dirs = [Path(p) for p in args.snapshots_dir]
  import importlib.resources

  snapshot_dirs.extend(
    [
      Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
      Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
      Path("../ml-framework-snapshots/snapshots"),
      Path("../ml-framework-snapshots"),
      Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
      Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
      Path("../ml-switcheroo-ir"),
      Path("src/ml_switcheroo/semantics"),
    ]
  )
  for pkg in ("ml_framework_snapshots.snapshots", "ml_ecosystem_snapshots.snapshots"):
    try:
      snap_res = importlib.resources.files(pkg)
      snapshot_dirs.append(Path(str(snap_res)))
    except (ImportError, ModuleNotFoundError, TypeError, OSError):
      pass
  snapshots = load_snapshots_multi(snapshot_dirs)

  grounding_engine = None
  if GroundingEngine is not None:
    try:
      ge_dirs = [str(d) for d in snapshot_dirs if d.exists()]
      grounding_engine = GroundingEngine(base_dirs=ge_dirs)
    except (OSError, ValueError, TypeError, RuntimeError):
      grounding_engine = None

  print(f"Loaded {len(snapshots)} snapshots.")
  if args.strict:
    if not snapshots:
      print("\n❌ No snapshots were loaded in --strict mode.")
      return 1
    if args.framework and args.framework not in snapshots:
      print(f"\n❌ Requested framework '{args.framework}' not found in loaded snapshots.")
      return 1

  errors = audit_frameworks(mgr, snapshots, framework=args.framework, grounding_engine=grounding_engine)

  src_dirs = [Path("src/ml_switcheroo")]
  ast_errors = audit_python_ast(src_dirs, snapshots)
  errors.extend(ast_errors)

  snippet_errors = audit_inline_snippets(mgr, snapshots)
  errors.extend(snippet_errors)

  if args.report_path:
    checksums = compute_snapshot_checksums(snapshot_dirs)
    report = generate_audit_report(mgr, snapshots, errors, checksums=checksums)
    with open(args.report_path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"Saved audit report to {args.report_path}")

  if errors:
    print(f"\n❌ Found {len(errors)} mismatches:\n")
    for error in errors:
      print(f"  - {error}")
    if args.strict:
      return 1
  else:
    print("\n✅ All mapped APIs and arguments verified against snapshots.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
