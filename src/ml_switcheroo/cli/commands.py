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

# Re-export dependent classes to satisfy test patches that target this module
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.discovery.syncer import FrameworkSyncer
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.semantics.autogen import SemanticPersister
from ml_switcheroo.discovery.consensus import ConsensusEngine
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir

__all__ = [
  "BatchValidator",
  "ConsensusEngine",
  "FrameworkSyncer",
  "ReadmeEditor",
  "SemanticPersister",
  "SemanticsManager",
  "_capture_framework",
  "_convert_single_file",
  "_get_pkg_version",
  "_print_batch_summary",
  "_save_snapshot",
  "available_frameworks",
  "get_adapter",
  "handle_audit",
  "handle_ci",
  "handle_convert",
  "handle_docs",
  "handle_gen_tests",
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
