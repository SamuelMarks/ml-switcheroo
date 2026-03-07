"""
Discovery & Standards Command Handlers.

This module implements the logic for populating the Semantic Knowledge Base via
Automated Discovery and Specification Imports.

It handles:
1.  **Ingestion**: Importing external standards (ONNX, Array API).
2.  **Scaffolding**: Generating skeleton mappings.
3.  **Discovery**: Identifying Neural Layers via Consensus and linking them to the Semantic Hub.
4.  **StableHLO**: Importing StableHLO specificiation from markdown.
5.  **SASS**: Importing NVIDIA Instruction Sets from HTML documentation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.frameworks.base import (
  StandardCategory,
  get_adapter,
)
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter
from ml_switcheroo.importers.stablehlo_reader import StableHloSpecImporter
from ml_switcheroo.importers.sass_reader import SassSpecImporter
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.frameworks.loader import get_definitions_path
from ml_switcheroo.utils.console import (
  console,
  log_error,
  log_info,
  log_success,
  log_warning,
)


def handle_scaffold(frameworks: list[str], out_dir: Path) -> int:
  """
  Handles the 'scaffold' command.

  -   Iterates through provided frameworks.
  -   Uses `FrameworkAdapter.discovery_heuristics` regexes to fuzzy match framework
      APIs against known specs or structural conventions.
  -   Populates semantic specs (Hub) and framework mappings (Spokes/Snapshots).

  Args:
      frameworks (list[str]): List of framework package names to scaffold (e.g., ['torch', 'jax']).
      out_dir (Path): The root directory for generating the knowledge base.
          Defaults to the package source if None.

  Returns:
      int: Exit code.
  """
  semantics = SemanticsManager()  # pragma: no cover
  scaffolder = Scaffolder(semantics=semantics)  # pragma: no cover
  scaffolder.scaffold(frameworks, root_dir=out_dir)  # pragma: no cover
  return 0  # pragma: no cover


def handle_import_spec(target: Path) -> int:
  """
  Handles the 'import-spec' command.

  Parses upstream standards documentation or stubs and merges the definitions
  into the local semantics Hub.

  Args:
      target (Path): .md file, .html file, or directory to parse.

  Returns:
      int: Exit code.
  """
  out_dir = resolve_semantics_dir()  # pragma: no cover
  out_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover

  # 1. Parsing Logic for Markdown Files
  if target.is_file() and target.suffix == ".md":  # pragma: no cover
    # Distinguish between ONNX and StableHLO based on content header
    try:  # pragma: no cover
      content_header = target.read_text(encoding="utf-8", errors="ignore")[:300]  # pragma: no cover
    except Exception as e:  # pragma: no cover
      log_error(f"Failed to read file header: {e}")  # pragma: no cover
      return 1  # pragma: no cover

    if "StableHLO" in content_header or "stablehlo" in content_header.lower():  # pragma: no cover
      log_info(f"Detected StableHLO Spec: {target.name}")  # pragma: no cover
      importer = StableHloSpecImporter()  # pragma: no cover
      data = importer.parse_file(target)  # pragma: no cover
      _save_spec(out_dir, "k_stablehlo.json", data)  # pragma: no cover
      return 0  # pragma: no cover
    else:
      log_info(f"Detected ONNX Markdown Spec: {target.name}")  # pragma: no cover
      importer = OnnxSpecImporter()  # pragma: no cover
      data = importer.parse_file(target)  # pragma: no cover
      _save_spec(out_dir, "k_neural_net.json", data)  # pragma: no cover
      return 0  # pragma: no cover

  # 2. Parsing Logic for HTML Files (NVIDIA SASS)
  elif target.is_file() and target.suffix == ".html":  # pragma: no cover
    log_info(f"Detected SASS HTML Spec: {target.name}")  # pragma: no cover
    importer = SassSpecImporter()  # pragma: no cover
    data = importer.parse_file(target)  # pragma: no cover

    # Special Case: SASS mappings go to frameworks/definitions/sass.json, NOT semantics/
    sass_def_path = get_definitions_path("sass")  # pragma: no cover

    # We reuse _save_spec logic but point it to the definitions directory
    _save_spec(sass_def_path.parent, sass_def_path.name, data)  # pragma: no cover
    return 0  # pragma: no cover

  # 3. Array API Stubs (Tier A)
  elif target.is_dir():  # pragma: no cover
    log_info("Detected Array API Stubs Directory")  # pragma: no cover
    importer = ArrayApiSpecImporter()  # pragma: no cover
    data = importer.parse_folder(target)  # pragma: no cover
    _save_spec(out_dir, "k_array_api.json", data)  # pragma: no cover
    return 0  # pragma: no cover

  else:
    log_error(
      "Invalid input. Must be .md (ONNX/StableHLO), .html (SASS), or dir of stubs (Array API)."
    )  # pragma: no cover
    return 1  # pragma: no cover


def _save_spec(out_dir: Path, filename: str, data: Dict[str, Any]) -> None:
  """
  Helper to persist JSON spec data.

  Args:
      out_dir: Target directory path.
      filename: JSON filename.
      data: Dictionary content to save.
  """
  out_p = out_dir / filename  # pragma: no cover
  final_data = data  # pragma: no cover

  if out_p.exists():  # pragma: no cover
    try:  # pragma: no cover
      with open(out_p, "rt", encoding="utf-8") as f:  # pragma: no cover
        existing = json.load(f)  # pragma: no cover
      log_info(f"Merging {filename} with existing {len(existing)} entries...")  # pragma: no cover
      existing.update(data)  # pragma: no cover
      final_data = existing  # pragma: no cover
    except Exception as e:  # pragma: no cover
      log_warning(f"Could not load existing {filename}: {e}. Overwriting.")  # pragma: no cover

  if not out_dir.exists():  # pragma: no cover
    out_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover

  with open(out_p, "wt", encoding="utf-8") as f:  # pragma: no cover
    json.dump(final_data, f, indent=2, sort_keys=True)  # pragma: no cover
  log_success(f"Saved {len(final_data)} entries to [path]{out_p}[/path]")  # pragma: no cover


def handle_sync_standards(categories: List[str], frameworks: Optional[List[str]], dry_run: bool) -> int:
  """
  Handles 'sync-standards' command.

  Invokes the Consensus Engine to automatically discover new Abstract Operations
  by finding commonalities across multiple framework API surfaces. Writes the
  results to `k_discovered.json` (The Unofficial/Discovered Standard).

  Args:
      categories (List[str]): List of category strings (layer, loss, etc.) to scan.
      frameworks (Optional[List[str]]): List of framework keys to scan.
          Defaults to installed/registered frameworks.
      dry_run (bool): If True, prints results without saving.

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

  # 3. Load Existing Semantics for Deduplication
  # We must not re-discover operations that are already defined in the Specs
  # (e.g. from ONNX or Array API).
  semantics_mgr = SemanticsManager()
  known_ops = set(semantics_mgr.get_known_apis().keys())

  engine = ConsensusEngine()
  persister = SemanticPersister()
  out_dir = resolve_semantics_dir()

  # MERGED LOGIC: Use 'k_discovered.json' as the persistent store for self-repaired standards.
  # This aligns with the 'handle_sync' command which consumes this specific file.
  target_file = out_dir / "k_discovered.json"

  total_persisted = 0

  for cat in valid_categories:
    log_info(f"Scanning category: [cyan]{cat.value}[/cyan]")
    framework_inputs: Dict[str, List[GhostRef]] = {}

    # A. Collect API surfaces
    for fw in frameworks:
      adapter = get_adapter(fw)
      if not adapter:
        continue  # pragma: no cover

      try:
        refs = adapter.collect_api(cat)
        if refs:
          framework_inputs[fw] = refs
          console.print(f"  - {fw}: Found {len(refs)} items")
      except Exception as e:  # pragma: no cover
        log_warning(f"Failed to collect {cat} from {fw}: {e}")  # pragma: no cover

    if len(framework_inputs) < 2:
      console.print("  [dim]Skipping consensus (need input from at least 2 frameworks)[/dim]")
      continue

    # B. Cluster & Align
    candidates = engine.cluster(framework_inputs)
    # Filter: Must be present in at least 2 frameworks to form a consensus standard
    common = engine.filter_common(candidates, min_support=2)

    # --- DEDUPLICATION ---
    # Extract only candidates that are NOT currently known
    new_candidates = []
    skipped_count = 0
    for cand in common:
      if cand.name in known_ops:
        skipped_count += 1  # pragma: no cover
      else:
        new_candidates.append(cand)
        known_ops.add(cand.name)  # Add to set so we don't duplicate within this run

    if skipped_count > 0:
      console.print(f"  [dim]Skipped {skipped_count} candidates already defined in Specs.[/dim]")  # pragma: no cover

    if not new_candidates:
      continue  # pragma: no cover

    # Augment signatures on surviving candidates
    engine.align_signatures(new_candidates)

    console.print(f"  => Discovered {len(new_candidates)} new candidates (e.g. {new_candidates[0].name})")

    # C. Persist
    if dry_run:
      for c in new_candidates:  # pragma: no cover
        console.print(f"    [Dry] {c.name} (std_args={c.std_args})")  # pragma: no cover
    else:
      persister.persist(new_candidates, target_file)
      total_persisted += len(new_candidates)

  if not dry_run:
    if total_persisted > 0:
      log_success(f"Sync complete. Persisted {total_persisted} standards to {target_file.name}")
    else:
      log_info("Sync complete. No new standards found.")

  return 0
