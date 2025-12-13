"""
Convert Command Handler.

This module implements the logic for the `ml_switcheroo convert` command.
It orchestrates:
1. Configuration loading (including external plugin discovery).
2. Semantics initialization.
3. AST transformation via the Engine.
4. Verification harness generation (optional).
5. Output writing and trace logging.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.utils.console import (
  console,
  log_info,
  log_success,
  log_error,
  log_warning,
)
from rich.table import Table


def handle_convert(
  input_path: Path,
  output_path: Optional[Path],
  source: Optional[str],
  target: Optional[str],
  verify: bool,
  strict: Optional[bool],
  plugin_settings: Dict[str, Any],
  json_trace_path: Optional[Path] = None,
) -> int:
  """
  Handles the 'convert' command execution.

  Orchestrates the loading of configuration, initialization of the semantic
  knowledge base, and the execution of the transpilation engine on files or directories.

  Args:
      input_path: Path to the source file or directory to convert.
      output_path: Path where generated code should be saved.
      source: Override for source framework (e.g. 'torch').
      target: Override for target framework (e.g. 'jax').
      verify: If True, generates and runs a verification harness test immediately.
      strict: If True, enforces strict strict_mode on the Engine.
      plugin_settings: Dictionary of specific plugin configuration flags.
      json_trace_path: Optional path to dump execution trace JSON.

  Returns:
      int: Exit code (0 for success, 1 for failure).
  """
  if not input_path.exists():
    log_error(f"Input not found: {input_path}")
    return 1

  # 1. Load Configuration (TOML + CLI overrides)
  config = RuntimeConfig.load(
    source=source,
    target=target,
    strict_mode=strict,
    plugin_settings=plugin_settings,
    search_path=input_path if input_path.is_dir() else input_path.parent,
  )

  # 2. Wire External Plugins
  # If the user defined 'plugin_paths' in pyproject.toml, load them now so hooks are active.
  if config.plugin_paths:
    loaded_count = load_plugins(extra_dirs=config.plugin_paths)
    if loaded_count > 0:
      log_info(f"Loaded {loaded_count} external plugins from configuration.")

  semantics = SemanticsManager()
  batch_results: Dict[str, ConversionResult] = {}

  # 3. Process Input (File vs Directory)
  if input_path.is_file():
    result = _convert_single_file(input_path, output_path, semantics, verify, config, json_trace_path)
    batch_results[input_path.name] = result
    if not result.success:
      return 1

  elif input_path.is_dir():
    if not output_path:
      log_error("Directory conversion requires --out destination directory.")
      return 1

    py_files = list(input_path.rglob("*.py"))
    if not py_files:
      log_warning(f"No .py files found in {input_path}")
      return 0

    log_info(f"Processing {len(py_files)} files from {input_path}...")

    for src_file in py_files:
      rel_path = src_file.relative_to(input_path)
      dest_file = output_path / rel_path

      batch_trace = None
      if json_trace_path:
        # If doing a directory batch, we cannot write all traces to one file.
        # Heuristic: if trace path provided, write side-by-side with output?
        # Or simply allow trace naming derived from output structure.
        if output_path:
          batch_trace = (output_path / rel_path).with_suffix(".trace.json")

      result = _convert_single_file(src_file, dest_file, semantics, verify, config, batch_trace)
      batch_results[str(rel_path)] = result

  _print_batch_summary(batch_results)
  return 0


def _convert_single_file(
  input_path: Path,
  output_path: Optional[Path],
  semantics: SemanticsManager,
  verify: bool,
  config: RuntimeConfig,
  json_trace_path: Optional[Path] = None,
) -> ConversionResult:
  """
  Helper to execute transpilation logic on a single file.

  Args:
      input_path: Source file path.
      output_path: Destination file path.
      semantics: Loaded Semantics Manager.
      verify: Whether to run verification.
      config: Runtime configuration object.
      json_trace_path: Path to save trace event logs.

  Returns:
      ConversionResult: Result object containing status and code.
  """
  try:
    with open(input_path, "rt", encoding="utf-8") as f:
      code = f.read()
    engine = ASTEngine(semantics, config=config)
    result = engine.run(code)

    if json_trace_path and result.trace_events:
      try:
        json_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_trace_path, "wt", encoding="utf-8") as f:
          json.dump(result.trace_events, f, indent=2)
        log_info(f"Trace saved to [path]{json_trace_path}[/path]")
      except Exception as e:
        log_error(f"Failed to write trace: {e}")

    if not result.success:
      return result

    effective_out = output_path
    if verify and not effective_out:
      # If verify requested but no output, default to a temp-like name next to source
      effective_out = input_path.with_name(f"{input_path.stem}_converted.py")

    if output_path:
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "wt", encoding="utf-8") as f:
        f.write(result.code)
      log_success(f"Transpiled: [path]{input_path}[/path] -> [path]{output_path}[/path]")
    else:
      # Print to stdout if no output
      print(result.code)

    if verify and effective_out:
      log_info(f"Verifying {effective_out.name}...")
      harness_gen = HarnessGenerator()
      harness_path = effective_out.parent / f"verify_{effective_out.stem}.py"
      harness_gen.generate(
        source_file=input_path,
        target_file=effective_out,
        output_harness=harness_path,
        source_fw=config.source_framework,
        target_fw=config.target_framework,
      )
      proc = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True)
      if proc.returncode == 0:
        print("   ✨ Verification Passed")
      else:
        print(f"   ❌ Verification Failed (See {harness_path})")
        # Attach verification error to result so batch summary sees it
        result.errors.append("Verification Harness Failed")

    return result
  except Exception as e:
    log_error(f"Failed to convert {input_path}: {e}")
    return ConversionResult(success=False, errors=[str(e)])


def _print_batch_summary(results: Dict[str, ConversionResult]) -> None:
  """
  Renders a summary table of conversion results to the console.

  Args:
      results: Dictionary mapping filenames to conversion results.
  """
  total = len(results)
  successes = sum(1 for r in results.values() if r.success and not r.has_errors)
  failures = sum(1 for r in results.values() if not r.success or r.has_errors)

  if failures == 0:
    log_success(f"Batch Complete: {successes}/{total} files converted perfectly.")
    return

  table = Table(title="Transpilation Report")
  table.add_column("File", style="cyan")
  table.add_column("Status", justify="center")
  table.add_column("Issues", style="red")

  for filename, res in results.items():
    if res.success and not res.has_errors:
      continue
    status = "❌ Failed" if not res.success else "⚠️ Warnings"
    issues = "; ".join(res.errors) if res.errors else "Unknown Error"
    table.add_row(filename, status, issues)

  console.print(table)
  console.print(f"\n[bold]Summary:[/bold] {successes} Passed, {failures} with Issues.")
