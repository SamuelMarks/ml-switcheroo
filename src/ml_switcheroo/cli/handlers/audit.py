"""
Audit Command Handler.

Performs static analysis on source files to identify unsupported operations
and generate coverage reports.
"""

import json
from pathlib import Path
from typing import List, Set, Dict, Tuple
from rich.table import Table

import libcst as cst
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.analysis.audit import CoverageScanner
from ml_switcheroo.utils.console import log_info, log_error, console
from ml_switcheroo.frameworks import get_adapter


def resolve_roots(framework_keys: List[str]) -> Set[str]:
  """
  Expands framework keys (e.g. 'flax_nnx') to python module roots (e.g. 'flax', 'jax').

  This ensures that querying for 'flax_nnx' correctly finds 'import flax'.
  It consults the Adapter metadata for import aliases and search modules.

  Args:
      framework_keys: List of requested framework identifiers.

  Returns:
      A set of root module names (e.g. {'torch', 'jax', 'flax'}).
  """
  roots = set(framework_keys)

  for key in framework_keys:
    adapter = get_adapter(key)
    if not adapter:
      continue

    # 1. Import Alias (Primary Root)
    if hasattr(adapter, "import_alias") and adapter.import_alias:
      root = adapter.import_alias[0].split(".")[0]
      roots.add(root)

    # 2. Search Modules (Secondary Roots)
    # e.g. FlaxNNXAdapter scans 'jax.numpy' too, so 'jax' is added.
    if hasattr(adapter, "search_modules"):
      for mod in adapter.search_modules:
        root = mod.split(".")[0]
        roots.add(root)

  return roots


def handle_audit(path: Path, source_frameworks: List[str], json_mode: bool = False) -> int:
  """
  Scans a directory/file to determine coverage against the Knowledge Base.

  Args:
      path: Input source file or directory.
      source_frameworks: List of framework keys to scan for.
      json_mode: If True, output JSON to stdout and suppress Rich logs.

  Returns:
      int: Exit code (0 if audit reveals full coverage, 1 if missing ops).
           Note: In audit mode, missing ops might be considered a 'failure'.
  """
  if not path.exists():
    # Errors go to logs (stderr usually) but we want to fail cleanly.
    log_error(f"Path not found: {path}")
    return 1

  files = [path] if path.is_file() else list(path.rglob("*.py"))

  semantics = SemanticsManager()

  # Resolve roots implies expanding 'flax_nnx' to 'flax', 'torch' to 'torch', etc.
  allowed_roots = resolve_roots(source_frameworks)

  if not json_mode:
    log_info(f"Auditing {len(files)} files against roots: {list(allowed_roots)}...")

  # Aggregators: FQN -> (IsSupported, Framework)
  global_results: Dict[str, Tuple[bool, str]] = {}

  for f in files:
    try:
      code = f.read_text("utf-8")
      tree = cst.parse_module(code)
      scanner = CoverageScanner(semantics, allowed_roots)
      tree.visit(scanner)

      # Merge
      global_results.update(scanner.results)

    except Exception as e:
      # We log parse errors even in JSON mode to stderr as they are critical warnings
      log_error(f"Failed to parse {f.name}: {e}")

  # Partition results
  missing_ops = {k: v for k, v in global_results.items() if not v[0]}
  supported_ops = {k: v for k, v in global_results.items() if v[0]}

  if json_mode:
    output_list = []
    # Sort for deterministic output
    sorted_keys = sorted(global_results.keys())
    for op in sorted_keys:
      is_supported, fw = global_results[op]
      item = {
        "api": op,
        "supported": is_supported,
        "framework": fw,
      }
      if not is_supported:
        item["suggestion"] = "Run 'scaffold' or 'wizard'"
      output_list.append(item)

    # Print pure JSON to stdout
    print(json.dumps(output_list, indent=2))
    return 1 if missing_ops else 0

  # Render Table
  if missing_ops:
    table = Table(title="❌ Missing Operations (Not in Knowledge Base)")
    table.add_column("Framework", style="cyan")
    table.add_column("API Name", style="red")
    table.add_column("Suggestion", style="dim")

    # Sort by Framework then API Name
    sorted_keys = sorted(missing_ops.keys(), key=lambda k: (missing_ops[k][1], k))

    for op in sorted_keys:
      _, fw = missing_ops[op]
      suggestion = "Run 'scaffold' or 'wizard'"
      table.add_row(fw, op, suggestion)

    console.print(table)
    console.print("\n")

  # Summary Stats
  total = len(global_results)
  count_supp = len(supported_ops)
  count_miss = len(missing_ops)
  percent = (count_supp / total * 100) if total > 0 else 0

  console.print(f"[bold]Audit Summary for {path.name}[/bold]")
  console.print(f"Unique APIs Found: {total}")
  console.print(f"Supported:         [green]{count_supp}[/green]")
  console.print(f"Missing:           [red]{count_miss}[/red]")
  console.print(f"Coverage:          [blue]{percent:.1f}%[/blue]")

  return 1 if missing_ops else 0
