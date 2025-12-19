"""
Interactive Wizard for Semantic Mapping Discovery.
Updated to support Distributed Semantics (Split Specs/Mappings).

This module provides the `MappingWizard` class, an interactive CLI tool used
by developers to categorize unmapped APIs found in a source framework.
It guides the user through:

1.  Identifying unknown APIs.
2.  Assigning them to a Semantic Tier (Math, Neural, Extras).
3.  Normalizing arguments (e.g. renaming `dim` to `axis`).
4.  Defining the target mapping (Hub/Spoke split).
5.  Persisting changes to the JSON knowledge base.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.utils.console import console, log_info, log_success


class MappingWizard:
  """
  Interactive tool to build robust semantic mappings.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the Wizard.

    Args:
        semantics (SemanticsManager): The loaded semantics manager.
    """
    self.semantics = semantics
    self.console = console

  def start(self, package_name: str) -> None:
    """
    Starts the interactive session.

    Scans the target package for APIs not currently present in the
    semantics manager, then iterates through them prompting the user
    for categorization and mapping.

    Args:
        package_name (str): The python package to inspect (e.g., 'torch').
    """
    log_info(f"Scanning [code]{package_name}[/code] for unmapped APIs...")

    inspector = ApiInspector()
    catalog = inspector.inspect(package_name)

    missing = self._find_unmapped_apis(catalog)
    total = len(missing)

    if total == 0:
      log_success(f"No missing mappings found in {package_name}!")
      return

    self.console.print(
      Panel(
        f"[bold]Found {total} unmapped APIs.[/bold]\n"
        "Splitting definitions into Specs (semantics/) and Mappings (snapshots/)\n"
        "Press [red]Ctrl+C[/red] to exit anytime.",
        title="ml-switcheroo Contextual Wizard",
        style="cyan",
      )
    )

    completed = 0
    skipped = 0

    try:
      for api_path, details in sorted(missing.items()):
        self._render_card(api_path, details, completed, total)

        tier_choice = self._prompt_tier_decision()
        if tier_choice == "skip":
          skipped += 1
          self.console.print("")
          continue

        detected_args = details.get("detected_sig", details.get("params", []))
        std_args, source_arg_map = self._prompt_arg_normalization(detected_args, f"Source ({package_name})")

        target_variant = self._prompt_target_mapping(std_args)

        target_file = self._resolve_target_file(tier_choice)
        self._save_complex_entry(
          filename=target_file,
          api_path=api_path,
          doc_summary=details.get("doc_summary", ""),
          std_args=std_args,
          source_fw=package_name.split(".")[0],
          source_arg_map=source_arg_map,
          target_variant=target_variant,
        )
        completed += 1
        self.console.print("")

    except KeyboardInterrupt:
      self.console.print("\n[yellow]Wizard interrupted by user.[/yellow]")

    log_success(f"Session Complete. Mapped {completed} APIs (Skipped {skipped}).")

  def _find_unmapped_apis(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters the catalog for items not present in the semantics.

    Args:
        catalog (Dict): Results from ApiInspector.inspect().

    Returns:
        Dict: Subset of catalog containing only unmapped APIs.
    """
    known_apis = set()
    # Reverse index maps API paths (src) to abstracts
    if hasattr(self.semantics, "_reverse_index"):
      known_apis = set(self.semantics._reverse_index.keys())

    missing = {}
    for api_path, details in catalog.items():
      if api_path not in known_apis:
        missing[api_path] = details
    return missing

  def _render_card(self, api_path: str, details: Dict[str, Any], idx: int, total: int) -> None:
    """
    Renders a UI card with API details.

    Args:
        api_path (str): The fully qualified name (e.g. 'torch.abs').
        details (Dict): Metadata from inspector (signature, docstring).
        idx (int): Current index.
        total (int): Total items.
    """
    sig = str(details.get("detected_sig", details.get("params", [])))
    doc = details.get("doc_summary", "No documentation available.")
    if not doc:
      doc = "No documentation available."
    suggestion = details.get("suggested_tier", "Unknown")

    content = (
      f"[bold cyan]API:[/bold cyan] {api_path}\n"
      f"[bold]Signature:[/bold] {sig}\n"
      f"[bold]Suggestion:[/bold] {suggestion}\n\n"
      f"[dim]{doc}[/dim]"
    )
    self.console.print(Panel(content, title=f"Item {idx + 1}/{total}", border_style="blue"))

  def _prompt_tier_decision(self) -> str:
    """
    Prompts the user to categorize the operation.

    Returns:
        str: One of 'math', 'neural', 'extras', or 'skip'.
    """
    choices = ["[M]ath", "[N]eural", "[E]xtras", "[S]kip"]
    options_text = " / ".join(choices)
    while True:
      resp = Prompt.ask(
        f"Bucket ({options_text})",
        choices=["m", "n", "e", "s", "M", "N", "E", "S"],
        show_choices=False,
      ).lower()
      mapping = {"m": "math", "n": "neural", "e": "extras", "s": "skip"}
      return mapping[resp]

  def _prompt_arg_normalization(self, detected_args: List[str], ctx_label: str) -> tuple[List[str], Dict[str, str]]:
    """
    Prompts user to rename arguments to standard names.

    Args:
        detected_args (List[str]): List of argument names found in source.
        ctx_label (str): Label for the UI context (e.g. "Source (torch)").

    Returns:
        tuple: (List of Standard Args, Map {standard_arg -> source_arg}).
    """
    if not detected_args:
      return [], {}
    self.console.print(f"[dim]Normalizing arguments for {ctx_label}... (Press Enter to keep original)[/dim]")
    std_args = []
    mapping = {}
    for arg in detected_args:
      if arg == "self":
        continue
      new_name = Prompt.ask(f"  Standard name for '[bold]{arg}[/bold]'", default=arg)
      std_args.append(new_name)
      if new_name != arg:
        mapping[new_name] = arg
    return std_args, mapping

  def _prompt_target_mapping(self, std_args: List[str]) -> Optional[Dict[str, Any]]:
    """
    Prompts user to define the target framework implementation immediately.

    Args:
        std_args (List[str]): The standard argument names defined in the previous step.

    Returns:
        Optional[Dict]: Target mapping definition or None if skipped.
    """
    if not Confirm.ask("Map to a Target Framework (e.g. JAX) now?", default=False):
      return None
    target_fw = Prompt.ask("Target Framework", default="jax")
    target_api = Prompt.ask("Target API Path", default=f"{target_fw}.numpy.???")
    plugin_name = None
    if Confirm.ask("Does this mapping require a plugin (e.g. decompose)?", default=False):
      plugin_name = Prompt.ask("Plugin Trigger Name (e.g. 'decompose_alpha')")
    target_arg_map = {}
    if std_args:
      self.console.print(f"[dim]Map Standard Args to {target_fw} params...[/dim]")
      for std in std_args:
        tgt_arg = Prompt.ask(f"  {target_fw} param for '[bold]{std}[/bold]'", default=std)
        if tgt_arg != std:
          target_arg_map[std] = tgt_arg
    return {
      "framework": target_fw,
      "data": {
        "api": target_api,
        "args": target_arg_map if target_arg_map else None,
        "requires_plugin": plugin_name,
      },
    }

  def _resolve_target_file(self, choice: str) -> str:
    """Helper to map user choice to filename."""
    if choice == "math":
      return "k_array_api.json"
    if choice == "neural":
      return "k_neural_net.json"
    return "k_framework_extras.json"

  def _save_complex_entry(
    self,
    filename: str,
    api_path: str,
    doc_summary: str,
    std_args: List[str],
    source_fw: str,
    source_arg_map: Dict[str, str],
    target_variant: Optional[Dict[str, Any]],
  ) -> None:
    """
    Persists the wizard results to disk, splitting Spec and Mapping data.
    Writes to Spec file AND Snapshot files.

    Args:
        filename (str): Target semantics spec file (e.g. 'k_neural_net.json').
        api_path (str): The source API path (e.g. 'torch.nn.Linear').
        doc_summary (str): Documentation string.
        std_args (List[str]): List of standardized argument names.
        source_fw (str): Source framework key (e.g. 'torch').
        source_arg_map (Dict): Mapping for source arguments.
        target_variant (Optional[Dict]): Details for the optional target mapping.
    """
    sem_dir = resolve_semantics_dir()
    snap_dir = resolve_snapshots_dir()

    if not sem_dir.exists():
      sem_dir.mkdir(parents=True, exist_ok=True)
    if not snap_dir.exists():
      snap_dir.mkdir(parents=True, exist_ok=True)

    abstract_id = api_path.split(".")[-1]

    # 1. Update Spec (Semantics)
    spec_path = sem_dir / filename
    self._write_to_file(spec_path, abstract_id, {"description": doc_summary, "std_args": std_args})

    # 2. Update Source Mapping (Snapshot)
    source_mapping = {"api": api_path}
    if source_arg_map:
      source_mapping["args"] = source_arg_map

    self._write_to_snapshot(snap_dir, source_fw, abstract_id, source_mapping)

    # 3. Update Target Mapping (Snapshot)
    if target_variant:
      fw = target_variant["framework"]
      v_data = target_variant["data"]
      if not v_data.get("args"):
        v_data.pop("args", None)
      if not v_data.get("requires_plugin"):
        v_data.pop("requires_plugin", None)

      self._write_to_snapshot(snap_dir, fw, abstract_id, v_data)

    self.console.print(f"[green]Saved {abstract_id} to distributed mappings[/green]")

  def _write_to_file(self, path: Path, key: str, data: Dict) -> None:
    """Helper to read-update-write a JSON file."""
    current = {}
    if path.exists():
      try:
        with open(path, "r", encoding="utf-8") as f:
          current = json.load(f)
      except Exception:
        pass

    if key in current:
      current[key].update(data)
    else:
      current[key] = data

    with open(path, "w", encoding="utf-8") as f:
      json.dump(current, f, indent=2, sort_keys=True)

  def _write_to_snapshot(self, snap_dir: Path, fw: str, key: str, data: Dict) -> None:
    """Helper to write to framework-specific snapshot using 'latest' version."""
    path = snap_dir / f"{fw}_vlatest_map.json"
    current = {"__framework__": fw, "mappings": {}}

    if path.exists():
      try:
        with open(path, "r", encoding="utf-8") as f:
          current = json.load(f)
      except Exception:
        pass

    if "mappings" not in current:
      current["mappings"] = {}

    current["mappings"][key] = data

    with open(path, "w", encoding="utf-8") as f:
      json.dump(current, f, indent=2, sort_keys=True)

  def _save_entry(self, api_path: str, details: Dict[str, Any], filename: str) -> None:
    """
    Legacy compatibility wrapper for direct saving.

    Args:
        api_path: Source API string.
        details: Metadata dict.
        filename: Target spec file.
    """
    summary = details.get("doc_summary", "")
    sig = details.get("detected_sig", details.get("params", []))

    # infer source framework from api path prefix
    source_fw = api_path.split(".")[0]

    # Fix for test mock data which often implies 'pkg' is framework
    if not source_fw:
      source_fw = "unknown"

    self._save_complex_entry(
      filename=filename,
      api_path=api_path,
      doc_summary=summary,
      std_args=sig,
      source_fw=source_fw,
      source_arg_map={},
      target_variant=None,
    )
