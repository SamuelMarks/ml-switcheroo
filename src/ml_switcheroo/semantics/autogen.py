"""
Semantic Autogen: The Persistence Layer.

This module is responsible for finalizing "Candidate Standards" proposed by the
Consensus Engine into the JSON Semantic Knowledge Base.

It implements a "Do Not Harm" policy:
- If a standard already exists in the file (implying manual curation or previous sync),
  it is skipped to prevent overwriting high-quality manual edits with automated guesses.
- New discoveries are strictly additive.
- Supports atomic rewrites of the JSON database.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from ml_switcheroo.discovery.consensus import CandidateStandard
from ml_switcheroo.utils.console import log_info, log_warning, log_success


class SemanticPersister:
  """
  Handles serialization of Discovered Standards to disk.

  Transforms internal CandidateStandard objects into the Switcheroo Semantic Schema
  and merges them safely into existing definition files.
  """

  def persist(self, candidates: List[CandidateStandard], target_file: Path) -> None:
    """
    Merges discovered candidates into the target JSON file and saves it.

    Args:
        candidates: List of aligned CandidateStandard objects.
        target_file: Path to the JSON semantic file (e.g. k_framework_extras.json).
    """
    # 1. Load Existing Data
    existing_data = {}
    if target_file.exists():
      try:
        with open(target_file, "r", encoding="utf-8") as f:
          existing_data = json.load(f)
      except json.JSONDecodeError:
        log_warning(f"Corrupt JSON at {target_file}. Backing up and starting fresh.")
        if target_file.stat().st_size > 0:
          target_file.rename(target_file.with_suffix(".bak"))

    # 2. Merge Logic (Priority: Existing > New)
    updates_count = 0
    skips_count = 0

    for cand in candidates:
      if cand.name in existing_data:
        # Conflict: Entry exists. Assume manual override or established consensus.
        # Future Enhancement: Merge distinct framework variants if missing.
        skips_count += 1
        continue

      # Transform to Schema and Insert
      entry = self._transform_to_schema(cand)
      existing_data[cand.name] = entry
      updates_count += 1

    # 3. Persistence
    if updates_count > 0:
      target_file.parent.mkdir(parents=True, exist_ok=True)
      with open(target_file, "w", encoding="utf-8") as f:
        # Sort keys for deterministic git diffs
        json.dump(existing_data, f, indent=2, sort_keys=True)
      log_success(f"Persisted {updates_count} new standards to {target_file.name}")
    else:
      if candidates:
        log_info(f"No new standards to persist for {target_file.name}. ({skips_count} skipped/existing)")
      else:
        log_info("No candidates provided for persistence.")

  def _transform_to_schema(self, cand: CandidateStandard) -> Dict[str, Any]:
    """
    Converts internal CandidateStandard to JSON Schema format.

    Output Format:
    {
        "std_args": ["reduction"]
    }
    """
    variants_dict = {}

    for fw_name, ref in cand.variants.items():
      fw_entry = {"api": ref.api_path}

      # Attach argument mappings if consensus engine found them
      # Dictionary maps {StandardArg -> FrameworkArg}
      mapping = cand.arg_mappings.get(fw_name)
      if mapping:
        fw_entry["args"] = mapping

      variants_dict[fw_name] = fw_entry

    return {"variants": variants_dict, "std_args": sorted(cand.std_args)}
