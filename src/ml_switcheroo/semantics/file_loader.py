"""
File Loading Logic for Semantic Knowledge Base.

This module handles the discovery and deserialization of JSON specification files
(The Hub) and Snapshot overlays (The Spokes). It delegates the actual data
combination logic to `ml_switcheroo.semantics.merging`.

Updates:
- Removed early exit if semantics directory is missing, enabling Snapshots to load independently.
- Using explicit path resolution imports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.merging import (
  merge_tier_data,
  merge_overlay_data,
  infer_tier_from_priority,
)
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir

# Filenames to treat as discovered/consensus content
DISCOVERED_FILENAMES = {"k_discovered.json"}


class KnowledgeBaseLoader:
  """
  Handles the I/O operations for populating the SemanticsManager.
  """

  def __init__(self, manager: Any):
    """
    Initialize the loader.

    Args:
        manager: The parent SemanticsManager instance to populate.
    """
    self.mgr = manager

  def load_knowledge_graph(self) -> None:
    """
    Scans the semantics directory for JSON specifications.

    Loads files in priority order:
    1. Array API (Math) - Priority 10
    2. Neural Net (Layers) - Priority 20
    3. Discovered/Extras - Priority 30
    """
    base_path = resolve_semantics_dir()

    # FIX: Do not return early if semantics dir missing.
    # Overlays (Snapshots) might still exist and need loading.
    if base_path.exists():
      all_files = list(base_path.rglob("*.json"))
      prioritized_files: List[Tuple[int, Path]] = []

      for fpath in all_files:
        fname = fpath.name
        priority = 30
        if "array" in fname:
          priority = 10
        elif "neural" in fname:
          priority = 20
        elif fname in DISCOVERED_FILENAMES:
          priority = 20
        prioritized_files.append((priority, fpath))

      # Sort by priority
      prioritized_files.sort(key=lambda x: (x[0], x[1].name))

      for priority, fpath in prioritized_files:
        try:
          with open(fpath, "r", encoding="utf-8") as f:
            content = json.load(f)
          tier = infer_tier_from_priority(priority)
          self._load_tier_content(content, tier)
        except Exception as e:
          print(f"⚠️ Error loading {fpath.name}: {e}")

    # Load Overlays after specs (or even if specs missing)
    self._load_overlays()

  def _load_overlays(self) -> None:
    """
    Scans the snapshots directory for framework mapping overlays.
    """
    snap_dir = resolve_snapshots_dir()
    if not snap_dir.exists():
      return

    mapping_files = list(snap_dir.glob("*_map.json"))
    for fpath in mapping_files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)
        self._load_overlay_content(content, fpath.name)
      except Exception as e:
        print(f"⚠️ Error loading overlay {fpath.name}: {e}")

  def _load_tier_content(self, content: Dict[str, Any], tier: SemanticTier) -> None:
    """
    Merges a specification dictionary into the manager.

    Args:
        content: The JSON content.
        tier: The Semantic Tier classification.
    """
    merge_tier_data(
      data=self.mgr.data,
      key_origins=self.mgr._key_origins,
      framework_configs=self.mgr.framework_configs,
      new_content=content,
      tier=tier,
    )

  def _load_overlay_content(self, content: Dict[str, Any], filename: str) -> None:
    """
    Merges a snapshot overlay into the manager.

    Args:
        content: The JSON content.
        filename: The source filename (used for metadata inference).
    """
    merge_overlay_data(
      data=self.mgr.data,
      key_origins=self.mgr._key_origins,
      framework_configs=self.mgr.framework_configs,
      test_templates=self.mgr.test_templates,
      content=content,
      filename=filename,
    )
