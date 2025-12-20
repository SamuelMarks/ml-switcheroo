"""
Main Entry Point for ml-switcheroo CLI.

This module handles argument parsing and dispatches to specific command
handlers defined in `ml_switcheroo.cli.commands`.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ml_switcheroo.config import parse_cli_key_values
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.cli import commands
from ml_switcheroo import __version__


def main(argv: Optional[List[str]] = None) -> int:
  """
  Main CLI entry point.

  Parses arguments via argparse and calls the appropriate handler function.

  Args:
      argv: Optional list of command line arguments (defaults to sys.argv).

  Returns:
      int: Exit code (0 for success, non-zero for failure).
  """
  parser = argparse.ArgumentParser(description="ml-switcheroo: Deterministic AST Transpiler")
  parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

  subparsers = parser.add_subparsers(dest="command", required=True)

  # --- Command: AUDIT ---
  cmd_audit = subparsers.add_parser("audit", help="Check source code for unsupported operations")
  cmd_audit.add_argument("path", type=Path, help="Input source file or directory")
  cmd_audit.add_argument(
    "--roots",
    nargs="+",
    default=["torch", "flax_nnx", "tensorflow", "mlx", "keras"],
    help="Framework roots to scan for (default: torch flax_nnx tensorflow mlx keras)",
  )

  # --- Command: CONVERT ---
  cmd_conv = subparsers.add_parser("convert", help="Transpile a Python file or directory")
  cmd_conv.add_argument("path", type=Path, help="Input source file or directory")
  cmd_conv.add_argument("--source", default=None, help="Source framework (default: from toml)")
  cmd_conv.add_argument("--target", default=None, help="Target framework (default: from toml)")
  cmd_conv.add_argument("--out", type=Path, help="Output destination (file or dir)")
  cmd_conv.add_argument(
    "--verify",
    action="store_true",
    help="Generate and run a verification harness immediately after conversion",
  )
  cmd_conv.add_argument(
    "--strict",
    action="store_true",
    default=None,
    help="Fail on unknown APIs instead of passing them through (Overrides config)",
  )
  cmd_conv.add_argument(
    "--json-trace", type=Path, default=None, help="Dump full execution trace (events, diffs) to a JSON file."
  )
  cmd_conv.add_argument(
    "--config",
    nargs="*",
    help="Plugin configuration flags in key=value format (e.g. epsilon=1e-5 use_custom=True)",
  )

  # --- Command: MATRIX ---
  subparsers.add_parser("matrix", help="Show compatibility table")

  # --- Command: WIZARD (Interactive Discovery) ---
  cmd_wiz = subparsers.add_parser("wizard", help="Interactively categorize missing mappings")
  cmd_wiz.add_argument("package", help="Package to scan (e.g. torch)")

  # --- Command: HARVEST (Automatic Learning) ---
  cmd_harv = subparsers.add_parser("harvest", help="Extract valid mappings from Manual Tests")
  cmd_harv.add_argument("path", type=Path, help="Path to manual test file(s) or directory")
  cmd_harv.add_argument(
    "--target",
    default="jax",
    help="Target framework found in tests (default: jax)",
  )
  cmd_harv.add_argument(
    "--dry-run",
    action="store_true",
    help="Print updates without writing to disk",
  )

  # --- Command: CI (Validation & Readme & Lockfile) ---
  cmd_ci = subparsers.add_parser("ci", help="Run validation suite")
  cmd_ci.add_argument("--update-readme", action="store_true", help="Rewrite README.md with results")
  cmd_ci.add_argument("--readme-path", type=Path, default=Path("README.md"))
  cmd_ci.add_argument(
    "--json-report",
    type=Path,
    default=None,
    help="Save verification results to a JSON file (Lockfile)",
  )

  # --- Command: SNAPSHOT (Ghost Protocol) ---
  cmd_snap = subparsers.add_parser("snapshot", help="Capture API surfaces for Ghost Mode support")
  cmd_snap.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Output directory (Defaults to src/ml_switcheroo/snapshots)",
  )

  # --- Command: Scaffold ---
  cmd_scaf = subparsers.add_parser("scaffold", help="Auto-generate mappings for frameworks")
  cmd_scaf.add_argument("--frameworks", nargs="+", default=["torch", "jax"], help="List of frameworks")
  cmd_scaf.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="Root directory for Knowledge Base. Must contain/create 'semantics' and 'snapshots' subdirs. Default: Package source.",
  )

  # --- Command: GEN DOCS (Migration Guide) ---
  cmd_docs = subparsers.add_parser("gen-docs", help="Generate Migration Guide Markdown")
  cmd_docs.add_argument("--source", default="torch", help="Source framework (default: torch)")
  cmd_docs.add_argument("--target", default="jax", help="Target framework (default: jax)")
  cmd_docs.add_argument(
    "--out",
    type=Path,
    default=Path("MIGRATION_GUIDE.md"),
    help="Output markdown file",
  )

  # --- Command: IMPORT SPEC ---
  cmd_imp = subparsers.add_parser("import-spec", help="Parse Array API RST/Stubs into JSON")
  cmd_imp.add_argument("target", type=Path, help="File or Folder to parse")

  # --- Command: SYNC ---
  cmd_sync = subparsers.add_parser("sync", help="Link a framework to the Spec")
  cmd_sync.add_argument("framework", help="Framework to sync (e.g. torch, jax, keras)")

  # --- Command: SYNC-STANDARDS (Generic Consensus) ---
  cmd_cons = subparsers.add_parser("sync-standards", help="Discovers and amends Abstract Standards via Consensus")
  cmd_cons.add_argument("--frameworks", nargs="+", help="Frameworks to scan (default: all installed)")
  cmd_cons.add_argument(
    "--categories",
    nargs="+",
    default=["loss", "optimizer", "layer", "activation"],
    help="API Categories to scan (enum values)",
  )
  cmd_cons.add_argument(
    "--dry-run",
    action="store_true",
    help="Print candidates without writing to disk",
  )

  # --- Command: GEN TESTS ---
  cmd_gen = subparsers.add_parser("gen-tests", help="Generate physical Python test files")
  cmd_gen.add_argument("--out", type=Path, default=Path("tests", "generated", "test_tier_a_math.py"))

  args = parser.parse_args(argv)

  if args.command == "audit":
    return commands.handle_audit(args.path, args.roots)

  elif args.command == "convert":
    settings = parse_cli_key_values(args.config)
    return commands.handle_convert(
      args.path, args.out, args.source, args.target, args.verify, args.strict, settings, args.json_trace
    )

  elif args.command == "matrix":
    return commands.handle_matrix()

  elif args.command == "wizard":
    return commands.handle_wizard(args.package)

  elif args.command == "harvest":
    return commands.handle_harvest(args.path, args.target, args.dry_run)

  elif args.command == "ci":
    return commands.handle_ci(args.update_readme, args.readme_path, args.json_report)

  elif args.command == "snapshot":
    return commands.handle_snapshot(args.out_dir)

  elif args.command == "scaffold":
    return commands.handle_scaffold(args.frameworks, args.out_dir)

  elif args.command == "gen-docs":
    return commands.handle_docs(args.source, args.target, args.out)

  elif args.command == "import-spec":
    return commands.handle_import_spec(args.target)

  elif args.command == "sync":
    return commands.handle_sync(args.framework)

  elif args.command == "sync-standards":
    return commands.handle_sync_standards(args.categories, args.frameworks, args.dry_run)

  elif args.command == "gen-tests":
    return commands.handle_gen_tests(args.out)

  return 0


if __name__ == "__main__":
  sys.exit(main())
