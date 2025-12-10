"""
CLI Command Handlers.

This module contains the logic for executing specific sub-commands of the CLI.
It decouples argument handling from execution logic.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.discovery.updater import MappingsUpdater
from ml_switcheroo.cli.wizard import MappingWizard
from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.utils.doc_gen import MigrationGuideGenerator
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter
from ml_switcheroo.discovery.syncer import FrameworkSyncer
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.core.hooks import load_plugins
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
  """Handles the 'convert' command."""
  if not input_path.exists():
    log_error(f"Input not found: {input_path}")
    return 1

  config = RuntimeConfig.load(
    source=source,
    target=target,
    strict_mode=strict,
    plugin_settings=plugin_settings,
    search_path=input_path if input_path.is_dir() else input_path.parent,
  )
  semantics = SemanticsManager()

  batch_results: Dict[str, ConversionResult] = {}

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
        if output_path:
          batch_trace = (output_path / rel_path).with_suffix(".trace.json")

      result = _convert_single_file(src_file, dest_file, semantics, verify, config, batch_trace)
      batch_results[str(rel_path)] = result

  _print_batch_summary(batch_results)
  return 0


def handle_matrix() -> int:
  """Handles 'matrix' command."""
  semantics = SemanticsManager()
  matrix = CompatibilityMatrix(semantics)
  matrix.render()
  return 0


def handle_update(package: str, auto_merge: bool, report_path: Path) -> int:
  """Handles 'update' command."""
  semantics = SemanticsManager()
  updater = MappingsUpdater(semantics)
  updater.update_package(package, auto_merge=auto_merge, report_path=report_path)
  return 0


def handle_wizard(package: str) -> int:
  """Handles 'wizard' command."""
  semantics = SemanticsManager()
  wizard = MappingWizard(semantics)
  wizard.start(package)
  return 0


def handle_harvest(path: Path, target: str, dry_run: bool) -> int:
  """Handles 'harvest' command."""
  semantics = SemanticsManager()
  harvester = SemanticHarvester(semantics, target_fw=target)
  files = []
  if path.is_file():
    files.append(path)
  elif path.is_dir():
    files.extend(list(path.rglob("test_*.py")))
  else:
    log_error(f"Invalid path: {path}")
    return 1

  total_updated = 0
  for f in files:
    total_updated += harvester.harvest_file(f, dry_run=dry_run)

  if total_updated > 0 and not dry_run:
    log_success(f"Harvest complete. Updated {total_updated} definitions.")
  elif total_updated == 0:
    log_info("No new manual fixes found to harvest.")
  return 0


def handle_ci(update_readme: bool, readme_path: Path, json_report: Optional[Path]) -> int:
  """Handles 'ci' command."""
  try:
    config = RuntimeConfig.load()
    if config.plugin_paths:
      loaded = load_plugins(extra_dirs=config.plugin_paths)
      if loaded > 0:
        log_info(f"Loaded {loaded} external extensions for CI environment.")
  except Exception as e:
    log_warning(f"Could not load project config: {e}")

  semantics = SemanticsManager()
  log_info("Running Verification Suite...")
  validator = BatchValidator(semantics)

  manual_tests_dir = Path("tests")
  if not manual_tests_dir.exists():
    manual_tests_dir = None

  results = validator.run_all(verbose=True, manual_test_dir=manual_tests_dir)

  pass_count = sum(results.values())
  print(f"\n📊 Results: {pass_count}/{len(results)} mappings verified.")

  if update_readme:
    editor = ReadmeEditor(semantics, readme_path)
    editor.update_matrix(results)

  if json_report:
    try:
      json_report.parent.mkdir(parents=True, exist_ok=True)
      with open(json_report, "wt", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
      log_success(f"Verification report saved to [path]{json_report}[/path]")
    except Exception as e:
      log_error(f"Failed to save report: {e}")
      return 1
  return 0


def handle_scaffold(frameworks: list[str], out_dir: Path) -> int:
  """Handles 'scaffold' command."""
  semantics = SemanticsManager()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.scaffold(frameworks, out_dir)
  return 0


def handle_docs(source: str, target: str, out_path: Path) -> int:
  """Handles 'gen-docs' command."""
  semantics = SemanticsManager()
  log_info(f"Generating comparison: {source} -> {target} at {out_path}...")
  generator = MigrationGuideGenerator(semantics)
  markdown = generator.generate(source, target)
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(markdown)
  log_success(f"Documenation saved to [path]{out_path}[/path]")
  return 0


def handle_import_spec(target: Path) -> int:
  """Handles 'import-spec' command."""
  if target.is_file() and target.suffix == ".md":
    log_info(f"Detected ONNX Markdown Spec: {target.name}")
    importer = OnnxSpecImporter()
    data = importer.parse_file(target)
    out_json = "k_neural_net.json"

  elif target.is_dir():
    log_info("Detected Array API Stubs Directory")
    importer = ArrayApiSpecImporter()
    data = importer.parse_folder(target)
    out_json = "k_array_api.json"

  else:
    log_error("Invalid input. Must be .md (ONNX) or dir of .py stubs (Array API).")
    return 1

  out_dir = resolve_semantics_dir()
  out_dir.mkdir(parents=True, exist_ok=True)
  out_p = out_dir / out_json

  if out_p.exists():
    with open(out_p, "rt", encoding="utf-8") as f:
      existing = json.load(f)
    log_info(f"Merging with existing {len(existing)} entries...")
    existing.update(data)
    data = existing

  with open(out_p, "wt", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
  log_success(f"Saved {len(data)} operations to [path]{out_p}[/path]")
  return 0


def handle_sync(framework: str) -> int:
  """Handles 'sync' command."""
  out_dir = resolve_semantics_dir()
  syncer = FrameworkSyncer()

  # Update: Explicitly list all tier files to sync
  tiers = ["k_array_api.json", "k_neural_net.json", "k_framework_extras.json"]

  for filename in tiers:
    json_p = out_dir / filename

    if not json_p.exists():
      continue

    try:
      with open(json_p, "rt", encoding="utf-8") as f:
        data = json.load(f)

      friendly_name = filename.replace("k_", "").replace(".json", "")
      log_info(f"Syncing [{friendly_name}] tier for {framework}...")

      syncer.sync(data, framework)

      with open(json_p, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    except Exception as e:
      log_warning(f"Failed to sync {filename}: {e}")

  return 0


def handle_gen_tests(out: Path) -> int:
  """Handles 'gen-tests' command."""
  mgr = SemanticsManager()
  semantics = mgr.get_known_apis()
  out.parent.mkdir(parents=True, exist_ok=True)
  gen = TestGenerator()
  gen.generate(semantics, out)
  return 0


def _convert_single_file(
  input_path: Path,
  output_path: Optional[Path],
  semantics: SemanticsManager,
  verify: bool,
  config: RuntimeConfig,
  json_trace_path: Optional[Path] = None,
) -> ConversionResult:
  """Helper for converting a specific file."""
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
      effective_out = input_path.with_name(f"{input_path.stem}_converted.py")

    if output_path:
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "wt", encoding="utf-8") as f:
        f.write(result.code)
      log_success(f"Transpiled: [path]{input_path}[/path] -> [path]{output_path}[/path]")
    else:
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
        result.errors.append("Verification Harness Failed")

    return result
  except Exception as e:
    log_error(f"Failed to convert {input_path}: {e}")
    return ConversionResult(success=False, errors=[str(e)])


def _print_batch_summary(results: Dict[str, ConversionResult]) -> None:
  """Helper to print batch summary table."""
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
