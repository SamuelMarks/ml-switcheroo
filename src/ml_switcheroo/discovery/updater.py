"""
Discovery Module for updating Semantic Knowledge Base.

This module scans installed packages (like 'torch') for APIs that are NOT present
in the current Semantic Knowledge Base. It generates a gap analysis report and,
optionally, automatically merges these missing definitions into the distributed
Knowledge Graph (Specs in `semantics/`, Mappings in `snapshots/`).

Includes Rich-based reporting for better developer experience.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from rich.table import Table
from rich import box

from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.utils.console import console, log_info, log_success, log_warning


class MappingsUpdater:
  """
  Identifies and remediates gaps between installed packages and Semantic JSONs.

  Attributes:
      semantics (SemanticsManager): The loaded knowledge base.
      inspector (ApiInspector): Static analysis tool for scanning packages.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the updater.

    Args:
        semantics: Instance of SemanticsManager containing current known APIs.
    """
    self.semantics = semantics
    self.inspector = ApiInspector()

  def update_package(
    self,
    package_name: str,
    auto_merge: bool = False,
    report_path: Optional[Path] = None,
  ) -> Dict[str, Any]:
    """
    Scans a package, identifies missing mappings, and updates the knowledge base.

    Args:
        package_name: The library to scan (e.g., 'torch').
        auto_merge: If True, writes new entries directly to json files.
        report_path: If provided, writes the raw missing dict to this JSON path.

    Returns:
        Dict[str, Any]: The dictionary of missing APIs found.
    """
    # 1. Scan the live environment
    log_info(f"Scanning [code]{package_name}[/code] for unmapped APIs...")
    catalog = self.inspector.inspect(package_name)

    # 2. Find Gaps
    missing = self._find_unmapped_apis(catalog)

    if not missing:
      log_success(f"No missing mappings found in {package_name}!")
      return {}

    # 3. Present Report
    self._print_rich_report(missing, auto_merge)

    # 4. Handle Output (Merge or Report)
    if auto_merge:
      self._merge_to_disk(missing, package_name)

    if report_path:
      self._write_report(missing, report_path)

    return missing

  def _find_unmapped_apis(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diffs the inspected catalog against the loaded semantics.

    Returns:
        Dict of APIs that are not in the SemanticsManager.
    """
    missing = {}

    for api_path, details in catalog.items():
      # If the API is not in our reverse index, it's missing.
      if not self.semantics.get_definition(api_path):
        # Heuristic: Filter out obvious internals or huge dumps
        if "._" in api_path:
          continue

        missing[api_path] = {
          "detected_sig": details.get("params", []),
          "doc_summary": details.get("docstring_summary", ""),
          "suggested_tier": self._guess_tier(api_path),
        }

    return missing

  def _guess_tier(self, api_name: str) -> str:
    """
    Heuristic to assign a destination JSON file.

    Args:
        api_name: e.g. 'torch.nn.Linear' or 'torch.abs'.

    Returns:
        Filename string: 'k_neural_net.json' or 'k_array_api.json'.
    """
    parts = api_name.split(".")
    name = parts[-1]

    # Heuristic 1: Neural path indicators
    if ".nn." in api_name or ".layers." in api_name:
      return "k_neural_net.json"

    # Heuristic 2: Constants/Classes often look like PascalCase in Torch
    if name[0].isupper() and "_" not in name:
      return "k_neural_net.json"

    return "k_array_api.json"  # Default bucket

  def _print_rich_report(self, missing: Dict[str, Any], merged: bool):
    """Prints a styled table of findings."""
    action = "Merged" if merged else "Missing"
    color = "green" if merged else "yellow"

    table = Table(title=f"Gap Analysis Report ({len(missing)} entries)", box=box.ROUNDED)
    table.add_column("API Path", style="cyan")
    table.add_column("Tier Suggestion", style="magenta")
    table.add_column("Status", style=color)

    # Limit rows for sanity
    shown_count = 0
    limit = 50

    for api, details in sorted(missing.items()):
      if shown_count < limit:
        table.add_row(api, details["suggested_tier"], f"[{color}]{action}[/{color}]")
      shown_count += 1

    if len(missing) > limit:
      table.add_row("...", "...", f"... and {len(missing) - limit} more")

    console.print(table)

  def _merge_to_disk(self, missing: Dict[str, Any], package_name: str):
    """
    Writes the missing APIs to the distributed knowledge base.
    Specs -> semantics/
    Mappings -> snapshots/{fw}_mappings.json
    """
    semantics_dir = resolve_semantics_dir()
    snapshots_dir = resolve_snapshots_dir()

    if not semantics_dir.exists():
      semantics_dir.mkdir(parents=True, exist_ok=True)
    if not snapshots_dir.exists():
      snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Infer framework key from package name (e.g. torch -> torch)
    fw_key = package_name.split(".")[0].lower()

    # Data Structures
    spec_updates: Dict[str, Dict[str, Any]] = defaultdict(dict)  # filename -> ops
    mapping_updates: Dict[str, Dict[str, Any]] = defaultdict(dict)  # fw_key -> ops

    for api_path, details in sorted(missing.items()):
      target_spec_file = details["suggested_tier"]
      # Use leaf name as Abstract ID (e.g. torch.abs -> 'abs')
      abstract_id = api_path.split(".")[-1]

      # 1. Prepare Spec Entry (Description + Args)
      spec_entry = {"description": details["doc_summary"], "std_args": details["detected_sig"]}
      # Note: We rely on file loading logic to merge this safely later
      spec_updates[target_spec_file][abstract_id] = spec_entry

      # 2. Prepare Mapping Entry (Api + Args)
      mapping_entry = {
        "api": api_path,
        # Default mapping assumes framework arg name == std arg name
        "args": {p: p for p in details["detected_sig"]},
      }
      mapping_updates[fw_key][abstract_id] = mapping_entry

    # --- Write Specs ---
    for filename, new_defs in spec_updates.items():
      self._update_json_file(semantics_dir / filename, new_defs)

    # --- Write Mappings ---
    for fw, mappings in mapping_updates.items():
      mapping_file = snapshots_dir / f"{fw}_mappings.json"
      # Structure for overlay file
      overlay_content = {"__framework__": fw, "mappings": mappings}
      # Specialized merge for overlays
      self._update_overlay_file(mapping_file, overlay_content)

  def _update_json_file(self, path: Path, new_entries: Dict[str, Any]):
    """Standard merge for Spec files."""
    current_data = {}
    if path.exists():
      try:
        with open(path, "rt", encoding="utf-8") as f:
          current_data = json.load(f)
      except Exception as e:
        log_warning(f"Could not read {path.name}: {e}")

    # Merge: favor existing if present (manual curation wins)
    added = 0
    for key, val in new_entries.items():
      if key not in current_data:
        current_data[key] = val
        added += 1

    if added > 0:
      with open(path, "wt", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, sort_keys=True)
      log_success(f"Added {added} specs to [path]{path.name}[/path]")

  def _update_overlay_file(self, path: Path, new_content: Dict[str, Any]):
    """Merges content into an overlay file."""
    current_data = {"__framework__": new_content["__framework__"], "mappings": {}}

    if path.exists():
      try:
        with open(path, "rt", encoding="utf-8") as f:
          current_data = json.load(f)
      except Exception:
        pass

    # Merge mappings
    new_mappings = new_content.get("mappings", {})
    # Use update to overwrite or add. For discovery, usually we want to add.
    # If the key exists, existing discovery might likely be same or better?
    # Let's use setdefault logic to be safe for overlays too.

    added = 0
    for op, details in new_mappings.items():
      if op not in current_data["mappings"]:
        current_data["mappings"][op] = details
        added += 1

    if added > 0:
      with open(path, "wt", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, sort_keys=True)
      log_success(f"Added {added} mappings to [path]{path.name}[/path]")

  def _write_report(self, missing: Dict[str, Any], path: Path):
    """Dumps validation report to JSON."""
    with open(path, "w", encoding="utf-8") as f:
      json.dump(missing, f, indent=2)
    log_info(f"Report written to [path]{path}[/path]")
