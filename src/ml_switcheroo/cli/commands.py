"""
CLI Command Handlers Facade.

This module primarily re-exports handlers from `ml_switcheroo.cli.handlers`
to maintain backward compatibility with existing tests and imports.
"""

from ml_switcheroo.cli.handlers.audit import handle_audit
from ml_switcheroo.cli.handlers.convert import (
  handle_convert,
  _convert_single_file,
  _print_batch_summary,
)

# New export
from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.cli.handlers.discovery import (
  handle_scaffold,
  handle_import_spec,
  handle_sync_standards,
)
from ml_switcheroo.cli.handlers.snapshots import (
  handle_snapshot,
  handle_sync,
  _get_pkg_version,
  _capture_framework,
  _save_snapshot,
)
from ml_switcheroo.cli.handlers.learning import (
  handle_wizard,
  handle_harvest,
)
from ml_switcheroo.cli.handlers.verify import handle_ci
from ml_switcheroo.cli.handlers.dev import (
  handle_matrix,
  handle_docs,
  handle_gen_tests,
)
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from typing import Optional
from pathlib import Path

# Re-export dependent classes to satisfy test patches that target this module
from ml_switcheroo.discovery.syncer import FrameworkSyncer
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


# Helper wrapper for the new handler to maintain signature consistency in CLI dispatch
def handle_gen_weight_script(
  source_file: Path, out_script: Path, source_fw: Optional[str] = None, target_fw: Optional[str] = None
) -> int:
  """
  Handler for generating weight migration script.
  """
  config = RuntimeConfig.load(source=source_fw, target=target_fw)
  semantics = SemanticsManager()

  generator = WeightScriptGenerator(semantics, config)
  success = generator.generate(source_file, out_script)

  return 0 if success else 1


__all__ = [
  "_capture_framework",
  "_convert_single_file",
  "_get_pkg_version",
  "_print_batch_summary",
  "_save_snapshot",
  "handle_audit",
  "handle_ci",
  "handle_convert",
  "handle_docs",
  "handle_gen_tests",
  "handle_gen_weight_script",
  "handle_harvest",
  "handle_import_spec",
  "handle_matrix",
  "handle_scaffold",
  "handle_snapshot",
  "handle_sync",
  "handle_sync_standards",
  "handle_wizard",
  "resolve_semantics_dir",
  "resolve_snapshots_dir",
]
