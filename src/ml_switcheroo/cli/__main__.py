"""Main Entry Point for ml-switcheroo CLI.

This module handles argument parsing and dispatches to specific command
handlers defined in `ml_switcheroo.cli.commands` and `ml_switcheroo.cli.handlers`.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ml_switcheroo.config import parse_cli_key_values
from ml_switcheroo.cli import commands

# Import direct handler for new 'define' command
from ml_switcheroo.cli.handlers.meta import handle_schema
from ml_switcheroo.cli.handlers.suggest import handle_suggest
from ml_switcheroo import __version__


def main(argv: Optional[List[str]] = None) -> int:
  """Main CLI entry point.

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
    "--sharding",
    action="store_true",
    help="Enable distributed sharding semantics (e.g. QKV fusions, JAX sharding constraints)",
  )
  cmd_conv.add_argument("--custom-snapshot", type=Path, help="Path to custom snapshot schema override")
  cmd_conv.add_argument(
    "--strict",
    action="store_true",
    default=None,
    help="Fail on unknown APIs instead of passing them through (Overrides config)",
  )
  cmd_conv.add_argument(
    "--intermediate",
    default=None,
    choices=["mlir", "tikz"],
    help="Force round-trip through intermediate representation for verification",
  )
  cmd_conv.add_argument(
    "--json-trace", type=Path, default=None, help="Dump full execution trace (events, diffs) to a JSON file."
  )
  cmd_conv.add_argument(
    "--config",
    nargs="*",
    help="Plugin configuration flags in key=value format (e.g. epsilon=1e-5 use_custom=True)",
  )

  # --- Command: DEFINE ---
  cmd_wgt = subparsers.add_parser("gen-weight-script", help="Generate a checkpoint migration script.")
  cmd_wgt.add_argument("source_file", type=Path, help="Path to the model source code file.")
  cmd_wgt.add_argument("--out", type=Path, required=True, help="Output path for the generated script.")
  cmd_wgt.add_argument("--source", default=None, help="Source framework (e.g. torch).")
  cmd_wgt.add_argument("--target", default=None, help="Target framework (e.g. jax).")

  # --- Command: MATRIX ---
  subparsers.add_parser("matrix", help="Show compatibility table")

  # --- Command: SCHEMA ---
  subparsers.add_parser("schema", help="Export ODL JSON Schema for LLM prompts/validation.")

  # --- Command: SUGGEST ---
  cmd_sug = subparsers.add_parser("suggest", help="Suggest an operation implementation")
  cmd_sug.add_argument("api", help="API path to suggest")
  cmd_sug.add_argument("--out-dir", type=Path, default=None, help="Output directory")
  cmd_sug.add_argument("--batch-size", type=int, default=50, help="Batch size")

  # --- Command: CI ---
  cmd_ci = subparsers.add_parser("ci", help="Run validation suite")
  cmd_ci.add_argument("--update-readme", action="store_true", help="Rewrite README.md with results")
  cmd_ci.add_argument("--readme-path", type=Path, default=Path("README.md"))
  cmd_ci.add_argument(
    "--json-report",
    type=Path,
    default=None,
    help="Save verification results to a JSON file (Lockfile)",
  )
  cmd_ci.add_argument(
    "--repair",
    action="store_true",
    help="Automatically relax constraints (tolerances) for failing tests and update specs accordingly.",
  )

  # --- Command: SNAPSHOT (Ghost Protocol) ---
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
  cmd_gen = subparsers.add_parser("gen-tests", help="Generate physical Python test files")
  cmd_gen.add_argument("--out", type=Path, default=Path("tests", "generated", "test_tier_a_math.py"))

  args = parser.parse_args(argv)

  if args.command == "convert":
    settings = parse_cli_key_values(args.config)
    return commands.handle_convert(
      args.path,
      args.out,
      args.source,
      args.target,
      args.verify,
      args.strict,
      args.intermediate,
      settings,
      args.json_trace,
      args.sharding,
    )

  elif args.command == "gen-weight-script":
    return commands.handle_gen_weight_script(args.source_file, args.out, args.source, args.target)

  elif args.command == "matrix":
    return commands.handle_matrix()

  elif args.command == "schema":
    return handle_schema()  # pragma: no cover

  elif args.command == "suggest":
    return handle_suggest(args.api, out_dir=args.out_dir, batch_size=args.batch_size)

  elif args.command == "ci":
    return commands.handle_ci(args.update_readme, args.readme_path, args.json_report, args.repair)

  elif args.command == "gen-docs":
    return commands.handle_docs(args.source, args.target, args.out)

  elif args.command == "gen-tests":
    return commands.handle_gen_tests(args.out)

  return 0  # pragma: no cover


if __name__ == "__main__":
  sys.exit(main())  # pragma: no cover
