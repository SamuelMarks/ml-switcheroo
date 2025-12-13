"""
Discovery Command Handlers.

This module implements the CLI logic for populating the Semantic Knowledge Base.
It handles:
1.  **Ingestion**: Importing external standards (ONNX, Array API) and Internal Golden Sets.
2.  **Syncing**: Linking abstract operations to installed framework implementations.
3.  **Discovery**: Scaffolding new frameworks and harvesting manual overrides.
4.  **Wiring**: Applying hardcoded adapter logic (plugins, templates) during sync.
"""

import json
import importlib.metadata
from pathlib import Path
from typing import Optional, List, Dict, Any

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.semantics.standards_internal import INTERNAL_OPS
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.cli.wizard import MappingWizard
from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter
from ml_switcheroo.discovery.syncer import FrameworkSyncer
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.frameworks.base import (
  StandardCategory,
  get_adapter,
  GhostRef,
  FrameworkAdapter,
  SNAPSHOT_DIR as DEFAULT_SNAP_DIR,
)
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.utils.console import (
  console,
  log_info,
  log_success,
  log_error,
  log_warning,
)


def handle_wizard(package: str) -> int:
  """
  Handles the 'wizard' command for interactive mapping discovery.

  Args:
      package: The name of the python package to inspect (e.g., 'torch').

  Returns:
      int: Exit code (0 for success).
  """
  semantics = SemanticsManager()
  wizard = MappingWizard(semantics)
  wizard.start(package)
  return 0


def handle_harvest(path: Path, target: str, dry_run: bool) -> int:
  """
  Handles the 'harvest' command to learn mappings from manual tests.

  Args:
      path: File or directory containing python test files.
      target: The framework target used in the tests (e.g., 'jax').
      dry_run: If True, prints changes without writing to disk.

  Returns:
      int: Exit code (0 for success, 1 for path errors).
  """
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


def handle_snapshot(out_dir: Optional[Path]) -> int:
  """
  Handles the 'snapshot' command.

  Scans installed frameworks and dumps their API signatures to JSON files.
  These snapshots allow the engine to operate in "Ghost Mode" (e.g., in WebAssembly)
  without requiring the heavy frameworks to be installed.

  Args:
      out_dir: Optional custom output directory. Defaults to package/snapshots.

  Returns:
      int: Exit code.
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
  """
  Handles the 'scaffold' command.

  Automatically generates mapping skeletons based on fuzzy matching between
  framework APIs and the semantic spec rules.

  Args:
      frameworks: List of framework names to scaffold.
      out_dir: The root directory for generating the knowledge base.

  Returns:
      int: Exit code.
  """
  semantics = SemanticsManager()
  scaffolder = Scaffolder(semantics=semantics)
  scaffolder.scaffold(frameworks, root_dir=out_dir)
  return 0


def handle_import_spec(target: Path) -> int:
  """
  Handles the 'import-spec' command.

  Parses upstream standards documentation or stubs and merges the definitions
  into the local semantics Hub.

  Supported Targets:
  1. 'internal': Loads the Internal Golden Sets for Optimizers/Vision/State.
  2. '*.md': Parses ONNX Markdown specifications.
  3. Directory: Parses Python Array API stub files.

  Args:
      target: The file, directory, or keyword to import.

  Returns:
      int: Exit code.
  """
  # 1. Internal Standards (Tier C)
  if str(target) == "internal":
    log_info("Loading Internal Standards Spec (Optimizers, Vision, State)...")
    data = INTERNAL_OPS
    out_json = "k_framework_extras.json"

  # 2. ONNX Spec (Tier B)
  elif target.is_file() and target.suffix == ".md":
    log_info(f"Detected ONNX Markdown Spec: {target.name}")
    importer = OnnxSpecImporter()
    data = importer.parse_file(target)
    out_json = "k_neural_net.json"

  # 3. Array API Stubs (Tier A)
  elif target.is_dir():
    log_info("Detected Array API Stubs Directory")
    importer = ArrayApiSpecImporter()
    data = importer.parse_folder(target)
    out_json = "k_array_api.json"

  else:
    log_error("Invalid input. Must be 'internal', .md (ONNX), or dir of stubs (Array API).")
    return 1

  # Merge and Save
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
  Handles the 'sync' command.

  Links the Abstract Standards (Hub) to the concrete implementation in a
  specific framework. This process involves:
  1.  Reading the Specs from `semantics/`.
  2.  Introspecting the installed framework to find matching APIs.
  3.  Applying Adapter-specific manual wiring (plugins, templates).
  4.  Writing the results to the Snapshot Overlay (`snapshots/{fw}_mappings.json`).

  Args:
      framework: The framework key to sync (e.g. 'torch', 'jax').

  Returns:
      int: Exit code.
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

  # 3. Apply Manual Wiring (Plugins & Templates) from Adapter
  # This replaces the standalone scripts/wire_*.py logic
  adapter = get_adapter(framework)
  if adapter and hasattr(adapter, "apply_wiring"):
    try:
      # We assume the adapter follows the protocol signature
      # apply_wiring(snapshot: Dict) -> None
      adapter.apply_wiring(snapshot_data)
      log_info(f"Applied manual wiring rules for {framework}.")
      # Assume wiring modified the dict, treat as update found
      total_found += 1
    except Exception as e:
      log_warning(f"Wiring failed for {framework}: {e}")

  # 4. Write Snapshot
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

  Invokes the Consensus Engine to automatically discover new Abstract Operations
  by finding commonalities across multiple framework API surfaces.

  Args:
      categories: List of API categories to scan (e.g. 'loss', 'layer').
      frameworks: List of frameworks to use for consensus (default: all).
      dry_run: If True, prints candidates without saving.

  Returns:
      int: Exit code.
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

  # We use pending_standards.json as the persistent store for self-repaired standards.
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


# --- Helpers ---


def _get_pkg_version(package_name: str) -> str:
  """
  Retrieves installed package version string safely.

  Args:
      package_name: The pip package name.

  Returns:
      Version string or "unknown".
  """
  try:
    if package_name == "torch":
      import torch

      return torch.__version__
    return importlib.metadata.version(package_name)
  except Exception:
    return "unknown"


def _capture_framework(fw_name: str) -> Dict[str, Any]:
  """
  Runs API collection for a single framework.

  Delegates to the framework's Adapter to scan all standardized categories.
  Used by `handle_snapshot`.

  Args:
      fw_name: The framework identifier.

  Returns:
      Dictionary containing captured API signatures and version metadata.
  """
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
  """
  Writes snapshot data to a JSON file.

  File naming convention: `{fw_name}_v{version}.json`
  (e.g., `torch_v2.0.1.json`).

  Args:
      fw_name: Framework name.
      data: The snapshot dictionary.
      target_dir: Directory to save the file.
  """
  if not data:
    return

  safe_ver = data["version"].replace("+", "_").replace(" ", "_")
  filename = f"{fw_name}_v{safe_ver}.json"
  out_path = target_dir / filename

  target_dir.mkdir(parents=True, exist_ok=True)

  with open(out_path, "w", encoding="utf_8") as f:
    json.dump(data, f, indent=2)

  log_success(f"Saved snapshot: [path]{out_path}[/path]")
