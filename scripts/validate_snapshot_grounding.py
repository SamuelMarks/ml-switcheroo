"""Validates framework mappings against extracted snapshots.

Provides a unified entrypoint delegating to `scripts.audit_against_snapshots`
and the canonical `scripts.snapshots` grounding engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_against_snapshots import main
from scripts.snapshots.ast_auditor import (
  NVIDIA_SASS_KNOWN_INSTRUCTIONS,
  RDNA_KNOWN_INSTRUCTIONS,
  audit_frameworks,
  audit_inline_snippets,
  audit_python_ast,
  extract_api_calls,
)
from scripts.snapshots.loader import (
  GroundingEngine,
  _flatten_single_framework,
  _resolve_grounding_engine,
  compute_snapshot_checksums,
  load_snapshots,
  load_snapshots_multi,
)
from scripts.snapshots.reporter import generate_audit_report

__all__ = [
  "GroundingEngine",
  "NVIDIA_SASS_KNOWN_INSTRUCTIONS",
  "RDNA_KNOWN_INSTRUCTIONS",
  "_flatten_single_framework",
  "_resolve_grounding_engine",
  "audit_frameworks",
  "audit_inline_snippets",
  "audit_python_ast",
  "compute_snapshot_checksums",
  "extract_api_calls",
  "generate_audit_report",
  "load_snapshots",
  "load_snapshots_multi",
  "main",
]

if __name__ == "__main__":
  sys.exit(main())
