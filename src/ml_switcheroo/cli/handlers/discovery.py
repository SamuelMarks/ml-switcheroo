"""
Discovery & Standards Command Handlers.

This module implements the logic for populating the Semantic Knowledge Base via
Automated Discovery and Specification Imports.

It handles:
1.  **Ingestion**: Importing external standards (ONNX, Array API) and Internal Golden Sets (Hub).
2.  **Scaffolding**: Generating skeleton mappings.
3.  **Discovery**: Identifying Neural Layers via Consensus.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.semantics.standards_internal import INTERNAL_OPS
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.discovery.layers import LayerDiscoveryBot
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.frameworks.base import (
  StandardCategory,
  get_adapter,
  GhostRef,
)
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.utils.console import (
  console,
  log_info,
  log_success,
  log_error,
  log_warning,
)


def handle_discover_layers(dry_run: bool) -> int:
  """
  Handles 'discover-layers' command.

  Runs the LayerDiscoveryBot to find missing Neural Network Layers (e.g. Linear, Conv)
  in Torch and Flax environments and populates 'k_discovered.json'.

  Args:
      dry_run: If True, prints findings without writing to disk.

  Returns:
      int: Exit code.
  """
  bot = LayerDiscoveryBot()
  count = bot.run(dry_run=dry_run)
  return 0 if count >= 0 else 1


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


def handle_sync_standards(categories: List[str], frameworks: Optional[List[str]], dry_run: bool) -> int:
  """
  Handles 'sync-standards' command.

  Invokes the Consensus Engine to automatically discover new Abstract Operations
  by finding commonalities across multiple framework API surfaces.
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
