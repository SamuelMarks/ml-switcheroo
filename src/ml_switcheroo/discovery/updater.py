"""
Discovery Module for updating Semantic Knowledge Base.

This module scans installed packages (like 'torch') for APIs that are NOT present
in the current Semantic Knowledge Base. It generates a gap analysis report and,
optionally, automatically merges these missing definitions into the appropriate
JSON tier files (`k_array_api.json` vs `k_neural_net.json`).

Includes Rich-based reporting for better developer experience.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from rich.table import Table
from rich import box

from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir
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
        auto_merge: If True, writes new entries directly to k_*.json files.
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
      self._merge_to_disk(missing)

    if report_path:
      self._write_report(missing, report_path)

    return missing

  def _find_unmapped_apis(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diffs the inspected catalog against the loaded semantics.

    Args:
        catalog: Result from ApiInspector.inspect().

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
    # (ignoring the .nn check above which catches most layers)
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

    # Limit rows for sanity if huge
    shown_count = 0
    limit = 50

    for api, details in missing.items():
      if shown_count < limit:
        table.add_row(api, details["suggested_tier"], f"[{color}]{action}[/{color}]")
      shown_count += 1

    if len(missing) > limit:
      table.add_row("...", "...", f"... and {len(missing) - limit} more")

    console.print(table)

  def _merge_to_disk(self, missing: Dict[str, Any]):
    """
    Writes the missing APIs into the actual source JSON files.
    """
    output_dir = resolve_semantics_dir()
    if not output_dir.exists():
      output_dir.mkdir(parents=True, exist_ok=True)

    # Group by Target File
    grouped: Dict[str, Dict[str, Any]] = {
      "k_array_api.json": {},
      "k_neural_net.json": {},
    }

    for api_path, details in missing.items():
      target_file = details["suggested_tier"]
      # Fallback if guess is weird
      if target_file not in grouped:
        target_file = "k_array_api.json"

      # Use leaf name as Abstract ID (e.g. torch.abs -> 'abs')
      abstract_id = api_path.split(".")[-1]

      # Create the entry skeleton
      entry = {
        "description": details["doc_summary"],
        "std_args": details["detected_sig"],
        "variants": {
          # We assume the package name from the api path start is the framework
          api_path.split(".")[0]: {"api": api_path}
        },
      }

      grouped[target_file][abstract_id] = entry

    # Write to files
    for filename, new_entries in grouped.items():
      if not new_entries:
        continue

      file_path = output_dir / filename
      current_data = {}

      if file_path.exists():
        try:
          with open(file_path, "rt", encoding="utf-8") as f:
            current_data = json.load(f)
        except json.JSONDecodeError:
          log_warning(f"Could not read existing {filename}, starting fresh.")

      # Update - respect manual overrides by only adding new keys
      added_count = 0
      for key, val in new_entries.items():
        if key not in current_data:
          current_data[key] = val
          added_count += 1
        else:
          # Key exists: assume manual curation is better, skip overwrite
          pass

      if added_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
          json.dump(current_data, f, indent=2, sort_keys=True)
        log_success(f"Added {added_count} new entries to [path]{filename}[/path]")
      else:
        log_info(f"No new unique entries for {filename} (Updates skipped).")

  def _write_report(self, missing: Dict[str, Any], path: Path):
    """Dumps validation report to JSON."""
    with open(path, "w", encoding="utf-8") as f:
      json.dump(missing, f, indent=2)
    log_info(f"Report written to [path]{path}[/path]")
