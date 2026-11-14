"""Execute Entry Point for ml-switcheroo CLI.

This module handles argument parsing and dispatches to specific command
handlers defined in `ml_switcheroo.cli.commands` and `ml_switcheroo.cli.handlers`.
"""

import argparse
import json
import sys
import typing
from pathlib import Path

from ml_switcheroo import __version__
from ml_switcheroo.cli import commands
from ml_switcheroo.cli.handlers.completion import handle_completion
from ml_switcheroo.cli.handlers.harvest import handle_harvest
from ml_switcheroo.cli.handlers.import_onnx import handle_import_onnx
from ml_switcheroo.cli.handlers.meta import handle_schema
from ml_switcheroo.cli.handlers.scaffold import handle_scaffold
from ml_switcheroo.cli.handlers.suggest import handle_suggest
from ml_switcheroo.config import (
  ConversionMode,
  ExecutionBackend,
  OptimizationLevel,
  parse_cli_key_values,
)


def _build_parser() -> argparse.ArgumentParser:
  """Construct the top-level ``ml-switcheroo`` CLI ``ArgumentParser`` and subcommands.

  Returns:
      Configured ``argparse.ArgumentParser`` instance.
  """
  parser = argparse.ArgumentParser(description="ml-switcheroo: Deterministic AST Transpiler")
  parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

  subparsers = parser.add_subparsers(dest="command", required=True)

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
  cmd_conv.add_argument(
    "--strict",
    action="store_true",
    default=None,
    help="Fail on unknown APIs instead of passing them through (Overrides config)",
  )
  cmd_conv.add_argument(
    "--intermediate",
    default=None,
    choices=["mlir", "tikz", "ir", "ml_switcheroo_ir"],
    help="Force round-trip through an intermediate representation (choices: mlir, tikz, ir, ml_switcheroo_ir)",
  )
  cmd_conv.add_argument(
    "--json-trace", type=Path, default=None, help="Dump full execution trace (events, diffs) to a JSON file."
  )
  cmd_conv.add_argument(
    "--config",
    nargs="*",
    help="Plugin configuration flags in key=value format (e.g. epsilon=1e-5 use_custom=True)",
  )
  cmd_conv.add_argument(
    "--mode",
    default=None,
    choices=[m.value for m in ConversionMode],
    help="Conversion mode: 'preserve_trivia' (default, 1:1 whitespace and trivia preservation) or 'optimized' (multi-pass constant folding, CSE, DCE, fusion)",
  )
  cmd_conv.add_argument(
    "--optimize",
    action="store_true",
    default=False,
    help="Shorthand for --mode optimized (runs multi-pass compiler optimizations)",
  )
  cmd_conv.add_argument(
    "-O",
    "--opt-level",
    dest="opt_level",
    default=None,
    choices=[lvl.value for lvl in OptimizationLevel],
    help="Optimization level when running in optimized mode (default: O2)",
  )
  cmd_conv.add_argument(
    "--backend",
    "--execution-backend",
    dest="execution_backend",
    default=None,
    choices=[b.value for b in ExecutionBackend],
    help="Execution backend from ml-switcheroo-compiler (default: webgpu)",
  )
  cmd_conv.add_argument(
    "--on-device",
    dest="on_device",
    action="store_true",
    default=None,
    help="Execute forward and backward passes on the configured backend",
  )

  cmd_def = subparsers.add_parser("define", help="Inject a new ODL definition into the Knowledge Base")
  cmd_def.add_argument("path", type=Path, help="Path to the ODL YAML file")

  cmd_onnx = subparsers.add_parser(
    "import-onnx", help="Convert official ONNX markdown specifications into ODL YAML files"
  )
  cmd_onnx.add_argument("path", type=Path, help="Path to ONNX operator markdown file (e.g. Operators.md)")
  cmd_onnx.add_argument("--out-dir", type=Path, default=None, help="Output directory to save generated ODL YAML files")
  cmd_onnx.add_argument("--domain", default="ai.onnx", help="ONNX operator domain (default: ai.onnx)")
  cmd_onnx.add_argument("--opset-version", type=int, default=21, help="ONNX opset version (default: 21)")

  cmd_wgt = subparsers.add_parser("gen-weight-script", help="Generate a checkpoint migration script.")
  cmd_wgt.add_argument("source_file", type=Path, help="Path to the model source code file.")
  cmd_wgt.add_argument("--out", type=Path, required=True, help="Output path for the generated script.")
  cmd_wgt.add_argument("--source", default=None, help="Source framework (e.g. torch).")
  cmd_wgt.add_argument("--target", default=None, help="Target framework (e.g. jax).")

  subparsers.add_parser("matrix", help="Show compatibility table")
  subparsers.add_parser("schema", help="Export ODL JSON Schema for LLM prompts/validation.")

  cmd_sug = subparsers.add_parser("suggest", help="Suggest an operation implementation")
  cmd_sug.add_argument("api", help="API path to suggest")
  cmd_sug.add_argument("--out-dir", type=Path, default=None, help="Output directory")
  cmd_sug.add_argument("--batch-size", type=int, default=50, help="Batch size")

  cmd_scaff = subparsers.add_parser("scaffold", help="Scaffold an API mapping for a new library")
  cmd_scaff.add_argument("framework", help="Framework name (e.g. jax.numpy)")

  cmd_harv = subparsers.add_parser("harvest", help="Harvest mappings from manual tests")
  cmd_harv.add_argument("path", type=Path, help="Path to test file or directory")

  cmd_vp = subparsers.add_parser("verified-pipeline", help="Run the verified ingestion pipeline")
  cmd_vp.add_argument("path", type=Path, help="Input source file to run through pipeline")

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

  cmd_docs = subparsers.add_parser("gen-docs", help="Generate Migration Guide Markdown")
  cmd_docs.add_argument("--source", default="torch", help="Source framework (default: torch)")
  cmd_docs.add_argument("--target", default="jax", help="Target framework (default: jax)")
  cmd_docs.add_argument(
    "--out",
    type=Path,
    default=Path("MIGRATION_GUIDE.md"),
    help="Output markdown file",
  )

  cmd_gen = subparsers.add_parser("gen-tests", help="Generate physical Python test files")
  cmd_gen.add_argument("--out", type=Path, default=Path("tests", "generated", "test_tier_a_math.py"))

  cmd_comp = subparsers.add_parser("completion", help="Generate shell completion script (bash, zsh, fish)")
  cmd_comp.add_argument("shell", choices=["bash", "zsh", "fish"], help="Target shell type")

  return parser


def _dispatch_convert(args: argparse.Namespace) -> int:
  """Dispatch the ``convert`` CLI subcommand.

  Args:
      args: Parsed CLI argument namespace.

  Returns:
      Process exit code from ``commands.handle_convert``.
  """
  settings = parse_cli_key_values(args.config)
  opt_flag = bool(args.optimize)
  opt_lvl: str | None = args.opt_level
  exec_backend: str | None = args.execution_backend
  on_dev: bool | None = args.on_device
  resolved_mode: str | None = args.mode
  if opt_flag or (opt_lvl is not None and resolved_mode is None):
    resolved_mode = ConversionMode.OPTIMIZED.value
  return commands.handle_convert(
    commands.CliConvertOptions(
      input_path=args.path,
      output_path=args.out,
      source=args.source,
      target=args.target,
      verify=args.verify,
      strict=args.strict,
      intermediate=args.intermediate,
      plugin_settings=settings,
      json_trace_path=args.json_trace,
      enable_sharding=args.sharding,
      mode=resolved_mode,
      opt_level=str(opt_lvl) if isinstance(opt_lvl, str) else None,
      execution_backend=str(exec_backend) if isinstance(exec_backend, str) else None,
      enable_on_device_execution=bool(on_dev) if isinstance(on_dev, bool) else None,
    )
  )


def _dispatch_scaffold(args: argparse.Namespace) -> int:
  """Dispatch the ``scaffold`` CLI subcommand and return exit code 0.

  Args:
      args: Parsed CLI argument namespace.

  Returns:
      Exit code ``0``.
  """
  handle_scaffold(args)
  return 0


def _dispatch_harvest(args: argparse.Namespace) -> int:
  """Dispatch the ``harvest`` CLI subcommand and return exit code 0.

  Args:
      args: Parsed CLI argument namespace.

  Returns:
      Exit code ``0``.
  """
  handle_harvest(args)
  return 0


def _dispatch_verified_pipeline(args: argparse.Namespace) -> int:
  """Dispatch the ``verified-pipeline`` CLI subcommand.

  Args:
      args: Parsed CLI argument namespace.

  Returns:
      ``0`` when pipeline status is ``'success'``, otherwise ``1``.
  """
  from ml_switcheroo.ingestion.verified_pipeline import run_verified_pipeline

  with open(args.path, "r", encoding="utf-8") as f:
    result = run_verified_pipeline(f.read())
  print(json.dumps(result, indent=2))
  return 0 if result.get("status") == "success" else 1


def main(argv: list[str] | None = None) -> int:
  """Execute CLI entry point.

  Parses arguments via argparse and calls the appropriate handler function.

  Args:
      argv: Optional list of command line arguments (defaults to sys.argv).

  Returns:
      int: Exit code (0 for success, non-zero for failure).

  """
  parser = _build_parser()
  args = parser.parse_args(argv)

  handlers: dict[str, typing.Callable[[argparse.Namespace], int]] = {
    "convert": _dispatch_convert,
    "define": lambda a: commands.handle_define(a.path),
    "import-onnx": lambda a: handle_import_onnx(
      a.path,
      out_dir=a.out_dir,
      domain=a.domain,
      opset_version=a.opset_version,
    ),
    "gen-weight-script": lambda a: commands.handle_gen_weight_script(a.source_file, a.out, a.source, a.target),
    "matrix": lambda _a: commands.handle_matrix(),
    "schema": lambda _a: handle_schema(),
    "suggest": lambda a: handle_suggest(a.api, out_dir=a.out_dir, batch_size=a.batch_size),
    "scaffold": _dispatch_scaffold,
    "harvest": _dispatch_harvest,
    "verified-pipeline": _dispatch_verified_pipeline,
    "ci": lambda a: commands.handle_ci(a.update_readme, a.readme_path, a.json_report, a.repair),
    "gen-docs": lambda a: commands.handle_docs(a.source, a.target, a.out),
    "gen-tests": lambda a: commands.handle_gen_tests(a.out),
    "completion": lambda a: handle_completion(a.shell),
  }
  handler = handlers.get(str(args.command))
  return handler(args) if handler is not None else 0


if __name__ == "__main__":
  sys.exit(main())
