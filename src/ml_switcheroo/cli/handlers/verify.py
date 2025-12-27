"""CLI verification commands."""

import json
from pathlib import Path
from typing import Optional

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.utils.console import log_info, log_success, log_warning, log_error


def handle_ci(update_readme: bool, readme_path: Path, json_report: Optional[Path]) -> int:
  """Handles 'ci' command."""
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
