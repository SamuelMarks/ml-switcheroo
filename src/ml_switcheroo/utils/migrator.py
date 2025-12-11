"""
Semantic Migration Utility.

This module provides the `SemanticMigrator` tool designed to refactor the
Knowledge Base from a monolithic structure to a distributed "Hub-and-Spoke" model.

FUNCTIONALITY:
1.  **Reads** existing JSON files in `src/ml_switcheroo/semantics/`.
2.  **Extracts** implementation details (`variants`).
3.  **Extracts** testing configuration (`__templates__` or `k_test_templates.json`).
4.  **Splits** this data by framework (e.g., 'torch', 'jax').
5.  **Writes** framework-specific data to `src/ml_switcheroo/snapshots/{fw}_mappings.json`.
6.  **Cleans** the original semantic files, leaving only Abstract Specs (`std_args`, `description`).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from ml_switcheroo.semantics.manager import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.utils.console import console, log_info, log_success, log_warning


class SemanticMigrator:
  """
  Automates the migration of semantic definitions to the distributed overlay structure.

  Attributes:
      semantics_dir (Path): Source directory for Spec files.
      snapshots_dir (Path): Destination directory for Overlay files.
  """

  def __init__(self, semantics_path: Path = None, snapshots_path: Path = None):
    """
    Initializes the migrator.

    Args:
        semantics_path: Path to the semantics directory. Defaults to package location.
        snapshots_path: Path to the snapshots directory. Defaults to package location.
    """
    self.semantics_dir = semantics_path or resolve_semantics_dir()
    self.snapshots_dir = snapshots_path or resolve_snapshots_dir()

  def migrate(self, dry_run: bool = False) -> None:
    """
    Executes the migration process.

    1. Scans semantics/*.json.
    2. Aggregates all variants by framework.
    3. Writes snapshots/{fw}_mappings.json.
    4. Rewrites semantics/*.json (Spec-only).

    Args:
        dry_run: If True, prints actions without writing to disk.
    """
    if not self.semantics_dir.exists():
      log_warning(f"Semantics directory not found: {self.semantics_dir}")
      return

    # Data Stores
    # { filename: sanitized_content }
    new_specs: Dict[Path, Dict[str, Any]] = {}

    # { framework: { "mappings": {op: variant}, "templates": {...} } }
    framework_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"mappings": {}, "templates": {}})

    json_files = list(self.semantics_dir.glob("*.json"))
    log_info(f"Scanning {len(json_files)} semantic files...")

    # --- Phase 1: Scan & Extract ---
    for fpath in json_files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)
      except Exception as e:
        log_warning(f"Skipping {fpath.name} (Load Error): {e}")
        continue

      sanitized_content = {}

      # Handle Meta Blocks (Templates, Framework Configs)
      if "__templates__" in content:
        self._extract_templates(content.pop("__templates__"), framework_data)

      # Handle standalone template file (k_test_templates.json)
      if "test_templates" in fpath.name:
        self._extract_templates(content, framework_data)
        # Don't keep the file in specs if it's purely templates
        continue

      # Handle standard spec items
      for key, block in content.items():
        # Ignore other meta blocks for spec file
        if key.startswith("__"):
          sanitized_content[key] = block
          continue

        # Process Operation Definition
        spec_entry, variants_found = self._process_op(key, block)
        sanitized_content[key] = spec_entry

        # Aggregate Variants
        for fw, variant in variants_found.items():
          framework_data[fw]["mappings"][key] = variant

      new_specs[fpath] = sanitized_content

    # --- Phase 2: Write Overlays ---
    if not dry_run and not self.snapshots_dir.exists():
      self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    for fw, data in framework_data.items():
      overlay_file = self.snapshots_dir / f"{fw}_mappings.json"

      output_data = {"__framework__": fw, "mappings": data["mappings"]}
      if data["templates"]:
        output_data["templates"] = data["templates"]

      if dry_run:
        console.print(f"[dim]Would write:[/dim] {overlay_file.name} ({len(data['mappings'])} mappings)")
      else:
        with open(overlay_file, "w", encoding="utf-8") as f:
          json.dump(output_data, f, indent=2, sort_keys=True)
        log_success(f"Created overlay: [path]{overlay_file.name}[/path]")

    # --- Phase 3: Update Specs ---
    for fpath, content in new_specs.items():
      if dry_run:
        console.print(f"[dim]Would clean:[/dim] {fpath.name}")
      else:
        with open(fpath, "w", encoding="utf-8") as f:
          # Ensure consistent ordering
          json.dump(content, f, indent=2, sort_keys=True)
        log_info(f"Cleaned spec: [path]{fpath.name}[/path]")

    # --- Phase 4: Cleanup File Deletions ---
    # k_test_templates.json is no longer needed in semantics if migrated
    template_file = self.semantics_dir / "k_test_templates.json"
    if template_file.exists() and not dry_run:
      template_file.unlink()
      log_info(f"Removed legacy file: {template_file.name}")

  def _process_op(self, op_name: str, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Splits a single operation definition into Spec and Variants.

    Args:
        op_name: ID of the operation.
        data: The original dictionary block.

    Returns:
        (spec_dict, variants_dict)
    """
    spec = data.copy()
    variants = spec.pop("variants", {})

    # Clean Spec: Should only contain descriptive fields
    # (description, std_args, type, from)
    # variants are removed via pop above.

    return spec, variants

  def _extract_templates(self, templates: Dict[str, Any], framework_data: Dict[str, Dict]):
    """
    Merges test templates into the framework aggregation.
    """
    for fw, tmpl in templates.items():
      framework_data[fw]["templates"] = tmpl
