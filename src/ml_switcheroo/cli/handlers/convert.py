"""Convert Command Handler.

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
from typing import Optional, Dict, Any

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
  intermediate: Optional[str],
  plugin_settings: Dict[str, Any],
  json_trace_path: Optional[Path] = None,
  enable_sharding: bool = False,
) -> int:
  """Handles the 'convert' command execution.

  Orchestrates the loading of configuration, initialization of the semantic
  knowledge base, and the execution of the transpilation engine on files or directories.

  Args:
      input_path: Path to the source file or directory to convert.
      output_path: Path where generated code should be saved.
      source: Override for source framework (e.g. 'torch').
      target: Override for target framework (e.g. 'jax').
      verify: If True, generates and runs a verification harness test immediately.
      strict: If True, enforces strict strict_mode on the Engine.
      intermediate: If provided, forces a roundtrip via this IR (e.g. 'mlir').
      plugin_settings: Dictionary of specific plugin configuration flags.
      json_trace_path: Optional path to dump execution trace JSON.
      enable_sharding: If True, enables sharding extraction.

  Returns:
      int: Exit code (0 for success, 1 for failure).

  """
  if not input_path.exists():
    log_error(f"Input not found: {input_path}")  # pragma: no cover
    return 1  # pragma: no cover

  # 1. Load Configuration (TOML + CLI overrides)
  config = RuntimeConfig.load(
    source=source,
    target=target,
    strict_mode=strict,
    intermediate=intermediate,
    enable_sharding=enable_sharding,
    plugin_settings=plugin_settings,
    search_path=input_path if input_path.is_dir() else input_path.parent,
  )

  # 2. Wire External Plugins
  # If the user defined 'plugin_paths' in pyproject.toml, load them now so hooks are active.
  if config.plugin_paths:
    loaded_count = load_plugins(extra_dirs=config.plugin_paths)  # pragma: no cover
    if loaded_count > 0:  # pragma: no cover
      log_info(f"Loaded {loaded_count} external plugins from configuration.")  # pragma: no cover

  semantics = SemanticsManager()
  batch_results: Dict[str, ConversionResult] = {}

  # 3. Process Input (File vs Directory)
  if input_path.is_file():
    result = _convert_single_file(input_path, output_path, semantics, verify, config, json_trace_path)
    batch_results[input_path.name] = result
    if not result.success:
      return 1

  elif input_path.is_dir():  # pragma: no cover
    if not output_path:  # pragma: no cover
      log_error("Directory conversion requires --out destination directory.")  # pragma: no cover
      return 1  # pragma: no cover

    py_files = list(input_path.rglob("*.py"))  # pragma: no cover
    if not py_files:  # pragma: no cover
      log_warning(f"No .py files found in {input_path}")  # pragma: no cover
      return 0  # pragma: no cover

    log_info(f"Processing {len(py_files)} files from {input_path}...")  # pragma: no cover

    for src_file in py_files:  # pragma: no cover
      rel_path = src_file.relative_to(input_path)  # pragma: no cover
      dest_file = output_path / rel_path  # pragma: no cover

      batch_trace = None  # pragma: no cover
      if json_trace_path:  # pragma: no cover
        # If doing a directory batch, we cannot write all traces to one file.
        # Heuristic: if trace path provided, write side-by-side with output?
        # Or simply allow trace naming derived from output structure.
        if output_path:  # pragma: no cover
          batch_trace = (output_path / rel_path).with_suffix(".trace.json")  # pragma: no cover

      result = _convert_single_file(src_file, dest_file, semantics, verify, config, batch_trace)  # pragma: no cover
      batch_results[str(rel_path)] = result  # pragma: no cover

  _print_batch_summary(batch_results)  # pragma: no cover
  return 0  # pragma: no cover


def _convert_single_file(
  input_path: Path,
  output_path: Optional[Path],
  semantics: SemanticsManager,
  verify: bool,
  config: RuntimeConfig,
  json_trace_path: Optional[Path] = None,
) -> ConversionResult:
  """Helper to execute transpilation logic on a single file.

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
      try:  # pragma: no cover
        json_trace_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        with open(json_trace_path, "wt", encoding="utf-8") as f:  # pragma: no cover
          json.dump(result.trace_events, f, indent=2)  # pragma: no cover
        log_info(f"Trace saved to [path]{json_trace_path}[/path]")  # pragma: no cover
      except Exception as e:  # pragma: no cover
        log_error(f"Failed to write trace: {e}")  # pragma: no cover

    if not result.success:
      return result  # pragma: no cover

    effective_out = output_path
    if verify and not effective_out:
      # If verify requested but no output, default to a temp-like name next to source
      effective_out = input_path.with_name(f"{input_path.stem}_converted.py")  # pragma: no cover

    if output_path:
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "wt", encoding="utf-8") as f:
        f.write(result.code)  # pragma: no cover
      log_success(f"Transpiled: [path]{input_path}[/path] -> [path]{output_path}[/path]")  # pragma: no cover
    else:
      # Print to stdout if no output
      print(result.code)  # pragma: no cover

    if verify and effective_out:  # pragma: no cover
      log_info(f"Verifying {effective_out.name}...")  # pragma: no cover
      harness_gen = HarnessGenerator()  # pragma: no cover
      harness_path = effective_out.parent / f"verify_{effective_out.stem}.py"  # pragma: no cover
      harness_gen.generate(  # pragma: no cover
        source_file=input_path,
        target_file=effective_out,
        output_harness=harness_path,
        source_fw=config.source_framework,
        target_fw=config.target_framework,
      )
      proc = subprocess.run([sys.executable, str(harness_path)], capture_output=True, text=True)  # pragma: no cover
      if proc.returncode == 0:  # pragma: no cover
        print("   ✨ Verification Passed")  # pragma: no cover
      else:
        print(f"   ❌ Verification Failed (See {harness_path})")  # pragma: no cover
        # Attach verification error to result so batch summary sees it
        result.errors.append("Verification Harness Failed")  # pragma: no cover

    return result  # pragma: no cover
  except Exception as e:
    log_error(f"Failed to convert {input_path}: {e}")
    return ConversionResult(success=False, errors=[str(e)])


def _print_batch_summary(results: Dict[str, ConversionResult]) -> None:
  """Renders a summary table of conversion results to the console.

  Args:
      results: Dictionary mapping filenames to conversion results.

  """
  total = len(results)  # pragma: no cover
  successes = sum(1 for r in results.values() if r.success and not r.has_errors)  # pragma: no cover
  failures = sum(1 for r in results.values() if not r.success or r.has_errors)  # pragma: no cover

  if failures == 0:  # pragma: no cover
    log_success(f"Batch Complete: {successes}/{total} files converted perfectly.")  # pragma: no cover
    return  # pragma: no cover

  table = Table(title="Transpilation Report")  # pragma: no cover
  table.add_column("File", style="cyan")  # pragma: no cover
  table.add_column("Status", justify="center")  # pragma: no cover
  table.add_column("Issues", style="red")  # pragma: no cover

  for filename, res in results.items():  # pragma: no cover
    if res.success and not res.has_errors:  # pragma: no cover
      continue  # pragma: no cover
    status = "❌ Failed" if not res.success else "⚠️ Warnings"  # pragma: no cover
    issues = "; ".join(res.errors) if res.errors else "Unknown Error"  # pragma: no cover
    table.add_row(filename, status, issues)  # pragma: no cover

  console.print(table)  # pragma: no cover
  console.print(f"\n[bold]Summary:[/bold] {successes} Passed, {failures} with Issues.")  # pragma: no cover
