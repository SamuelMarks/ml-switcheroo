"""
CLI Command Handlers.

This module contains the logic for executing specific sub-commands of the CLI.
It decouples argument handling from execution logic.
"""

import sys
import json
import subprocess
import importlib.metadata
from pathlib import Path
from typing import Optional, Dict, Any, List

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.testing.harness_generator import HarnessGenerator
from ml_switcheroo.cli.matrix import CompatibilityMatrix
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
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.frameworks.base import (
  StandardCategory,
  get_adapter,
  GhostRef,
  FrameworkAdapter,
  SNAPSHOT_DIR as DEFAULT_SNAP_DIR,
)
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


def handle_snapshot(out_dir: Optional[Path]) -> int:
  """
  Handles 'snapshot' command.
  Scans installed frameworks and dumps API signatures to JSON.
  """
  target_dir = out_dir or DEFAULT_SNAP_DIR

  log_info(f"Starting Snapshot Capture to {target_dir}")

  frameworks = available_frameworks()

  if not frameworks:
    log_error("No frameworks registered in ml-switcheroo. Check installation.")
    return 1

  processed = 0
  for fw in frameworks:
    data = _capture_framework(fw)
    if data:
      _save_snapshot(fw, data, target_dir)
      processed += 1

  if processed == 0:
    log_warning("No snapshots were generated. Are Torch/Keras/JAX installed?")
    return 1

  log_success(f"Capture complete. Generated {processed} snapshot headers.")
  return 0


def handle_scaffold(frameworks: list[str], out_dir: Path) -> int:
  """Handles 'scaffold' command."""
  semantics = SemanticsManager()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.scaffold(frameworks, root_dir=out_dir)
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
  """
  Handles 'sync' command.

  Reads Abstract Specs from semantics/, scans target framework,
  and writes discovered implementation details to snapshots/{fw}_mappings.json.
  """
  # Resolve paths relative to package
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  # 1. Load or Initialize current Snapshot (to preserve other mappings)
  ver = _get_pkg_version(framework)
  snap_path = snap_dir / f"{framework}_v{ver}_map.json"
  snapshot_data = {"__framework__": framework, "mappings": {}}

  if snap_path.exists():
    try:
      with open(snap_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        if loaded.get("__framework__") == framework:
          snapshot_data = loaded
    except Exception as e:
      log_warning(f"Could not read existing snapshot at {snap_path}: {e}")

  # Existing entries we want to preserve/respect
  existing_mappings = snapshot_data.get("mappings", {})

  # 2. Run Syncer on Spec data
  syncer = FrameworkSyncer()
  tiers = ["k_array_api.json", "k_neural_net.json", "k_framework_extras.json"]

  total_found = 0

  for filename in tiers:
    spec_path = sem_dir / filename
    if not spec_path.exists():
      continue

    try:
      with open(spec_path, "r", encoding="utf-8") as f:
        tier_data = json.load(f)

      # Inject existing mappings into tier_data so Syncer can see them (and skip if needed)
      # syncer.sync only skips if `framework` is in `variants` keys
      for op_name, details in tier_data.items():
        if op_name in existing_mappings:
          if "variants" not in details:
            details["variants"] = {}
          details["variants"][framework] = existing_mappings[op_name]

      # Execute Discovery (Modifies tier_data in-place)
      syncer.sync(tier_data, framework)

      # Extract results
      count_in_tier = 0
      for op_name, details in tier_data.items():
        variants = details.get("variants", {})
        if framework in variants:
          # Update snapshot mapping definition
          snapshot_data["mappings"][op_name] = variants[framework]
          count_in_tier += 1

      if count_in_tier > 0:
        log_info(f"Scanned {filename}: Found {count_in_tier} entries.")
        total_found += count_in_tier

    except Exception as e:
      log_warning(f"Error processing {filename}: {e}")

  # 3. Write Snapshot
  if total_found > 0 or snap_path.exists():
    if not snap_dir.exists():
      snap_dir.mkdir(parents=True, exist_ok=True)

    with open(snap_path, "w", encoding="utf-8") as f:
      json.dump(snapshot_data, f, indent=2, sort_keys=True)

    log_success(f"Synced complete. Overlay updated at [path]{snap_path.name}[/path]")
  else:
    log_info(f"No mappings found/updated for {framework}.")

  return 0


def handle_sync_standards(categories: List[str], frameworks: Optional[List[str]], dry_run: bool) -> int:
  """
  Handles 'sync-standards' command.
  Invokes Consensus Engine to discover new abstract operations from framework APIs.
  """
  console.print("[bold blue]Starting Consensus Engine...[/bold blue]")

  # 1. Resolve Frameworks
  if not frameworks:
    frameworks = available_frameworks()

  # 2. Resolve Categories
  valid_categories = []
  for c in categories:
    try:
      valid_categories.append(StandardCategory(c.lower()))
    except ValueError:
      log_warning(f"Skipping unknown category: {c}")

  if not valid_categories:
    log_error("No valid categories to scan.")
    return 1

  engine = ConsensusEngine()
  persister = SemanticPersister()
  out_dir = resolve_semantics_dir()

  # We use pending_standards.json as the persistent store for self-repaired standards, forcing manual review.
  target_file = out_dir / "pending_standards.json"

  total_persisted = 0

  for cat in valid_categories:
    log_info(f"Scanning category: [cyan]{cat.value}[/cyan]")
    framework_inputs: Dict[str, List[GhostRef]] = {}

    # A. Collect API surfaces
    for fw in frameworks:
      adapter = get_adapter(fw)
      if not adapter:
        continue

      try:
        refs = adapter.collect_api(cat)
        if refs:
          framework_inputs[fw] = refs
          console.print(f"  - {fw}: Found {len(refs)} items")
      except Exception as e:
        log_warning(f"Failed to collect {cat} from {fw}: {e}")

    if len(framework_inputs) < 2:
      console.print("  [dim]Skipping consensus (need input from at least 2 frameworks)[/dim]")
      continue

    # B. Cluster & Align
    candidates = engine.cluster(framework_inputs)
    # Filter: Must be present in at least 2 frameworks to form a consensus standard
    common = engine.filter_common(candidates, min_support=2)

    if not common:
      continue

    # Augment signatures
    engine.align_signatures(common)

    console.print(f"  => Discovered {len(common)} candidates (e.g. {common[0].name})")

    # C. Persist
    if dry_run:
      for c in common:
        console.print(f"    [Dry] {c.name} (std_args={c.std_args})")
    else:
      persister.persist(common, target_file)
      total_persisted += len(common)

  if not dry_run:
    if total_persisted > 0:
      log_success(f"Sync complete. Persisted {total_persisted} standards to {target_file.name}")
    else:
      log_info("Sync complete. No new standards found.")

  return 0


def handle_gen_tests(out: Path) -> int:
  """Handles 'gen-tests' command."""
  mgr = SemanticsManager()
  semantics = mgr.get_known_apis()
  out.parent.mkdir(parents=True, exist_ok=True)
  gen = TestGenerator()
  gen.generate(semantics, out)
  return 0


# --- Helpers (Moved from Scripts) ---


def _get_pkg_version(package_name: str) -> str:
  """Retrieves installed package version string safely."""
  try:
    if package_name == "torch":
      import torch

      return torch.__version__
    return importlib.metadata.version(package_name)
  except Exception:
    return "unknown"


def _capture_framework(fw_name: str) -> Dict[str, Any]:
  """Runs API collection for a single framework."""
  adapter: FrameworkAdapter = get_adapter(fw_name)
  if not adapter:
    log_warning(f"Skipping {fw_name}: No adapter found.")
    return {}

  version = _get_pkg_version(fw_name)
  if version == "unknown":
    log_warning(f"Skipping {fw_name}: Library not installed in this environment.")
    return {}

  log_info(f"Scanning {fw_name} v{version}...")

  snapshot_data = {"version": version, "categories": {}}
  found_any = False

  # Iterate all standard categories defined in the Protocol
  for category in StandardCategory:
    try:
      # Polymorphic call to the adapter's implementation
      refs = adapter.collect_api(category)

      if refs:
        found_any = True
        snapshot_data["categories"][category.value] = [ref.model_dump(exclude_unset=True) for ref in refs]
        print(f"  - Found {len(refs)} {category.value} definitions.")
    except Exception as e:
      log_error(f"  FAILED collecting {category.value}: {e}")

  if not found_any:
    return {}

  return snapshot_data


def _save_snapshot(fw_name: str, data: Dict[str, Any], target_dir: Path):
  """Writes the snapshot data to JSON."""
  if not data:
    return

  safe_ver = data["version"].replace("+", "_").replace(" ", "_")
  filename = f"{fw_name}_v{safe_ver}.json"
  out_path = target_dir / filename

  target_dir.mkdir(parents=True, exist_ok=True)

  with open(out_path, "w", encoding="utf_8") as f:
    json.dump(data, f, indent=2)

  log_success(f"Saved snapshot: [path]{out_path}[/path]")


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
