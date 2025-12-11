#!/usr/bin/env python3
"""
Migration Script: Distribution of Semantics.

This script executes the refactoring of the Knowledge Base from monolithic
JSON files in `semantics/` to a distributed structure:
- Abstract Specs remain in `semantics/`.
- Framework Implementations (Variants) move to `snapshots/`.
- Testing Templates move to `snapshots/`.

Usage:
    python scripts/run_migration.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in python path to allow running from root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_switcheroo.utils.migrator import SemanticMigrator
from ml_switcheroo.utils.console import console


def main():
  parser = argparse.ArgumentParser(description="Migrate Semantics to Distributed Structure.")
  parser.add_argument("--dry-run", action="store_true", help="Simulate the migration without writing changes to disk.")
  parser.add_argument("--force", action="store_true", help="Proceed without confirmation.")

  args = parser.parse_args()

  # Resolve paths relative to the package location
  try:
    from ml_switcheroo.semantics.manager import resolve_semantics_dir, resolve_snapshots_dir

    sem_dir = resolve_semantics_dir()
    snap_dir = resolve_snapshots_dir()

    console.print(f"[bold cyan]Migration Plan:[/bold cyan]")
    console.print(f"  Source (Specs):    [blue]{sem_dir}[/blue]")
    console.print(f"  Target (Overlays): [blue]{snap_dir}[/blue]")

    if not args.dry_run and not args.force:
      user_input = input("\nThis will modify JSON files in your source tree. Continue? [y/N] ")
      if user_input.lower() != "y":
        console.print("[yellow]Aborted.[/yellow]")
        return

    migrator = SemanticMigrator(semantics_path=sem_dir, snapshots_path=snap_dir)
    migrator.migrate(dry_run=args.dry_run)

    if not args.dry_run:
      console.print("\n[bold green]Migration Complete.[/bold green]")
      console.print("Please verify the diff and run tests: `pytest tests`")

  except ImportError as e:
    console.print(f"[bold red]Error:[/bold red] Could not import package. Run from project root. {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
