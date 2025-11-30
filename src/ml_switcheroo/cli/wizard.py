"""
Interactive Wizard for Semantic Mapping Discovery.

This module provides the `MappingWizard`, a CLI tool that uses `rich` to
interactively iterate through unmapped APIs in a target package. It prompts
the user to:
1.  Classify each API into a Semantic Tier (Math, Neural, Extras).
2.  Refine the Abstract Standard Arguments (renaming implementation vars to standard vars).
3.  Optionally map a Target implementation (e.g., JAX) immediately.
4.  **Assign Plugins**: Identify if the mapping requires AST transformations (e.g. `decompose_alpha`).

This ensures that the JSON entries created are fully functional mappings,
not just empty stubs.
"""

import json
from typing import Dict, Any, List, Optional

from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from ml_switcheroo.semantics.manager import SemanticsManager, resolve_semantics_dir
from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.discovery.updater import MappingsUpdater
from ml_switcheroo.utils.console import console, log_info, log_success, log_warning


class MappingWizard:
  """
  Interactive tool to build robust semantic mappings.

  Attributes:
      semantics (SemanticsManager): The loaded knowledge base.
      console (Console): Rich console for rendering UI elements.
      _updater (MappingsUpdater): Logic to find missing APIs.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the Wizard.

    Args:
        semantics: The active SemanticsManager with current mappings logic.
    """
    self.semantics = semantics
    self.console = console  # Use global console for consistent theming
    self._updater = MappingsUpdater(semantics)

  def start(self, package_name: str) -> None:
    """
    Starts the interactive session.

    Scans the package, identifies gaps, and enters the prompt loop.

    Args:
        package_name: The name of the package to scan (e.g. 'torch').
    """
    log_info(f"Scanning [code]{package_name}[/code] for unmapped APIs...")

    # 1. Inspect
    inspector = ApiInspector()
    catalog = inspector.inspect(package_name)

    # 2. Diff
    missing = self._updater._find_unmapped_apis(catalog)
    total = len(missing)

    if total == 0:
      log_success(f"No missing mappings found in {package_name}!")
      return

    self.console.print(
      Panel(
        f"[bold]Found {total} unmapped APIs.[/bold]\n"
        "We will iterate through them. You can rename arguments to standards,\n"
        "assign plugins for complex logic, and map targets.\n"
        "Press [red]Ctrl+C[/red] to exit anytime.",
        title="ml-switcheroo Contextual Wizard",
        style="cyan",
      )
    )

    completed = 0
    skipped = 0

    # 3. Iterate
    try:
      for api_path, details in missing.items():
        self._render_card(api_path, details, completed, total)

        # A. Categorize
        tier_choice = self._prompt_tier_decision()
        if tier_choice == "skip":
          skipped += 1
          self.console.print("")
          continue

        # B. Define Abstract Interface (Args)
        # Robustly check detected_sig, then fallback to params check
        detected_args = details.get("detected_sig", details.get("params", []))

        std_args, source_arg_map = self._prompt_arg_normalization(detected_args, f"Source ({package_name})")

        # C. Optional Target Mapping (Now with Plugin Support)
        target_variant = self._prompt_target_mapping(std_args)

        # D. Save
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
        self.console.print("")  # Spacing

    except KeyboardInterrupt:
      self.console.print("\n[yellow]Wizard interrupted by user.[/yellow]")

    log_success(f"Session Complete. Mapped {completed} APIs (Skipped {skipped}).")

  def _render_card(self, api_path: str, details: Dict[str, Any], idx: int, total: int):
    """Displays a summary card for the current API."""
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

    self.console.print(
      Panel(
        content,
        title=f"Item {idx + 1}/{total}",
        border_style="blue",
      )
    )

  def _prompt_tier_decision(self) -> str:
    """Asks the user for categorization."""
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
    Allows user to rename detected arguments to standard names.
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
    Asks user if they want to define a JAX/Target mapping immediately.
    Includes support for defining required plugins.
    """
    # Default False prevents tests from hanging if unmocked
    if not Confirm.ask("Map to a Target Framework (e.g. JAX) now?", default=False):
      return None

    target_fw = Prompt.ask("Target Framework", default="jax")
    target_api = Prompt.ask("Target API Path", default=f"{target_fw}.numpy.???")

    # Plugin Prompt: Allows attaching complex logic (e.g. 'decompose_alpha')
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
    """Maps choice ID to filename."""
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
  ):
    """
    Writes the detailed entry to JSON.
    """
    out_dir = resolve_semantics_dir()
    if not out_dir.exists():
      out_dir.mkdir(parents=True, exist_ok=True)

    target_path = out_dir / filename

    # 1. Load existing
    current_data = {}
    if target_path.exists():
      try:
        with open(target_path, "rt", encoding="utf-8") as f:
          current_data = json.load(f)
      except json.JSONDecodeError:
        log_warning(f"Corrupt JSON in {filename}, starting fresh.")

    # 2. Construct Abstract ID
    abstract_id = api_path.split(".")[-1]

    # 3. Construct Source Variant
    source_data = {"api": api_path}
    if source_arg_map:
      source_data["args"] = source_arg_map

    # 4. Construct Entry
    entry = {
      "description": doc_summary,
      "std_args": std_args,
      "variants": {source_fw: source_data},
    }

    # 5. Merge Target Variant
    if target_variant:
      fw = target_variant["framework"]
      v_data = target_variant["data"]
      # Clean values to keep JSON tidy
      if not v_data.get("args"):
        v_data.pop("args", None)
      if not v_data.get("requires_plugin"):
        v_data.pop("requires_plugin", None)

      entry["variants"][fw] = v_data

    # 6. Merge into File Data
    if abstract_id in current_data:
      existing = current_data[abstract_id]
      existing["variants"].update(entry["variants"])
      if not existing.get("std_args"):
        existing["std_args"] = std_args
      current_data[abstract_id] = existing
    else:
      current_data[abstract_id] = entry

    # 7. Write
    with open(target_path, "w", encoding="utf-8") as f:
      json.dump(current_data, f, indent=2, sort_keys=True)

    self.console.print(f"[green]Saved {abstract_id} to {filename}[/green]")

  def _save_entry(self, api_path: str, details: Dict[str, Any], filename: str):
    """
    Legacy compatibility wrapper for existing tests.
    Maps the simple call to the new complex save logic.
    """
    summary = details.get("doc_summary", "")
    # Handle mismatch between 'detected_sig' and 'params' in mock data
    sig = details.get("detected_sig", details.get("params", []))

    self._save_complex_entry(
      filename=filename,
      api_path=api_path,
      doc_summary=summary,
      std_args=sig,
      source_fw=api_path.split(".")[0],
      source_arg_map={},
      target_variant=None,
    )
