"""
CLI verification commands.

Handles the execution of the verification suite (CI) and optional auto-repair logic.
"""

import json
from pathlib import Path
from typing import Optional

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.utils.console import log_info, log_success, log_warning, log_error


def handle_ci(update_readme: bool, readme_path: Path, json_report: Optional[Path], repair: bool = False) -> int:
  """
  Handles 'ci' command.

  Args:
      update_readme: If True, updates the compatibility matrix in the README.
      readme_path: Path to the README file.
      json_report: Optional path to dump results JSON.
      repair: If True, attempts to automatically relax tolerances for failing tests via bisection.

  Returns:
      int: Exit code (0 on success, 1 on error).
  """
  try:
    config = RuntimeConfig.load()
    if config.plugin_paths:
      loaded = load_plugins(extra_dirs=config.plugin_paths)
      if loaded > 0:
        log_info(f"Loaded {loaded} external extensions for CI environment.")
  except Exception as e:
    log_warning(f"Could not load project config: {e}")

  semantics = SemanticsManager()
  log_info("Running Verification Suite...")
  validator = BatchValidator(semantics)

  manual_tests_dir = Path("tests")
  if not manual_tests_dir.exists():
    manual_tests_dir = None

  results = validator.run_all(verbose=True, manual_test_dir=manual_tests_dir)

  # --- Auto-Repair Logic ---
  if repair:
    log_info("Starting Auto-Repair Bisection for failing operations...")
    bisector = SemanticsBisector(validator.runner)
    repaired_count = 0
    failures = {op for op, passed in results.items() if not passed}

    for op_name in failures:
      log_info(f"Attempting repair for '{op_name}'...")
      defn = semantics.get_definition_by_id(op_name)
      if not defn:
        continue

      patch = bisector.propose_fix(op_name, defn)
      if patch:
        semantics.update_definition(op_name, patch)
        results[op_name] = True
        repaired_count += 1
        log_success(f"Repaired '{op_name}' with new constraints.")
      else:
        log_warning(f"Could not repair '{op_name}'.")

    if repaired_count > 0:
      log_success(f"Auto-Repair completed. Fixed {repaired_count} operations.")
    else:
      log_info("Auto-Repair yielded no fixes.")

  pass_count = sum(results.values())
  print(f"\n📊 Results: {pass_count}/{len(results)} mappings verified.")

  if update_readme:
    editor = ReadmeEditor(semantics, readme_path)
    editor.update_matrix(results)

  if json_report:
    try:
      json_report.parent.mkdir(parents=True, exist_ok=True)
      with open(json_report, "wt", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
      log_success(f"Verification report saved to [path]{json_report}[/path]")
    except Exception as e:
      log_error(f"Failed to save report: {e}")
      return 1
  return 0
