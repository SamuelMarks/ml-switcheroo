"""
Scaffolding Tool for Knowledge Base Discovery.

This module provides the `Scaffolder` class, which inspects installed
libraries (Torch, JAX, etc.) and aligns them against the Specification-Guided
Knowledge Base.

It prioritizes matching discovered APIs against ingested Specs (ONNX, Array API)
before falling back to structural heuristics provided by Framework Adapters.

Updates for Distributed Semantics:
- Writes Abstract Operation Definitions (Specs) to `semantics/*.json` (Hub).
- Writes Implementation Details (Variants) to `snapshots/*.json` (Spokes).
"""

import json
import importlib.metadata
import difflib
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from collections import defaultdict

from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.utils.console import console, log_info, log_success
from ml_switcheroo.frameworks import get_adapter

# Access static definitions for common utilities
from ml_switcheroo.frameworks.common.data import get_dataloader_semantics


class Scaffolder:
  """
  Automated discovery tool that aligns framework APIs.

  Attributes:
      inspector (ApiInspector): Tool to extract function signatures.
      console (Console): Rich console for output.
      semantics (SemanticsManager): The knowledge base manager.
      similarity_threshold (float): Cutoff for fuzzy matching (0.0 - 1.0).
      arity_penalty (float): Penalty subtracted from similarity score for mismatched arity.
      _cached_heuristics (Dict[str, List[re.Pattern]]): Compiled regexes for categorization.
  """

  def __init__(
    self,
    semantics: Optional[SemanticsManager] = None,
    similarity_threshold: float = 0.8,
    arity_penalty: float = 0.3,
  ):
    self.inspector = ApiInspector()
    self.console = console
    self.semantics = semantics or SemanticsManager()
    self.similarity_threshold = similarity_threshold
    self.arity_penalty = arity_penalty

    # Staging areas
    self.staged_specs: Dict[str, Dict[str, Any]] = {
      "k_array_api.json": {},
      "k_neural_net.json": {},
      "k_framework_extras.json": {},
    }
    self.staged_mappings: Dict[str, Dict[str, Any]] = defaultdict(dict)
    self._cached_heuristics: Optional[Dict[str, List[re.Pattern]]] = None

  def _lazy_load_heuristics(self) -> Dict[str, List[re.Pattern]]:
    if self._cached_heuristics is not None:
      return self._cached_heuristics

    compiled: Dict[str, List[re.Pattern]] = {"neural": [], "extras": []}

    # We need a way to iterate available frameworks, importing from manager or frameworks package
    from ml_switcheroo.frameworks import available_frameworks

    for fw_name in available_frameworks():
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      if hasattr(adapter, "discovery_heuristics") and adapter.discovery_heuristics:
        heuristics = adapter.discovery_heuristics

        for tier, patterns in heuristics.items():
          if tier in compiled:
            for pat in patterns:
              try:
                compiled[tier].append(re.compile(pat, re.IGNORECASE))
              except re.error as e:
                print(f"⚠️ Invalid regex pattern in {fw_name} adapter: '{pat}' - {e}")

    self._cached_heuristics = compiled
    return compiled

  def scaffold(self, frameworks: List[str], root_dir: Optional[Path] = None):
    """
    Main entry point. Scans frameworks and builds/updates JSON mappings.

    Args:
        frameworks: List of package names (e.g. ['torch', 'jax']).
        root_dir: Root directory of the Knowledge Base (Hub & Spokes).
                  If provided, 'semantics/' and 'snapshots/' subdirs are used relative to this.
                  If None, defaults to package location.
    """
    if root_dir:
      semantics_path = root_dir / "semantics"
      snapshots_path = root_dir / "snapshots"
    else:
      semantics_path = resolve_semantics_dir()
      snapshots_path = resolve_snapshots_dir()

    # Pre-load known specs to drive categorization
    known_neural_ops = self._get_ops_by_tier(SemanticTier.NEURAL)
    known_math_ops = self._get_ops_by_tier(SemanticTier.ARRAY_API)
    known_extras_ops = self._get_ops_by_tier(SemanticTier.EXTRAS)

    catalogs = {}
    for fw in frameworks:
      log_info(f"Scanning [code]{fw}[/code]...")
      catalogs[fw] = self.inspector.inspect(fw)

    # Drive alignment by the first framework (Primary)
    primary_fw = "torch" if "torch" in frameworks else frameworks[0]
    primary_catalog = catalogs[primary_fw]

    log_info(f"Aligning APIs against [code]{primary_fw}[/code] and [code]Specs[/code]...")

    for api_path, details in primary_catalog.items():
      name = details["name"]
      kind = details.get("type", "function")

      # Strategy 1: Spec Validation (Priority)
      neural_match = self._match_spec_op(name, known_neural_ops)
      if neural_match:
        self._register_entry("k_neural_net.json", neural_match, primary_fw, api_path, details, catalogs)
        continue

      math_match = self._match_spec_op(name, known_math_ops)
      if math_match:
        self._register_entry("k_array_api.json", math_match, primary_fw, api_path, details, catalogs)
        continue

      extras_match = self._match_spec_op(name, known_extras_ops)
      if extras_match:
        self._register_entry("k_framework_extras.json", extras_match, primary_fw, api_path, details, catalogs)
        continue

      # Strategy 2: Heuristic Fallback (Dynamic)
      if self._is_structurally_neural(api_path, kind):
        # FIX: Ensure neural heuristic routes to neural spec, not extras
        self._register_entry("k_neural_net.json", name, primary_fw, api_path, details, catalogs)
        continue

      if self._is_structurally_extra(api_path, name):
        self._register_entry("k_framework_extras.json", name, primary_fw, api_path, details, catalogs)
        continue

      # Heuristic: Math Tier (Default)
      self._register_entry("k_framework_extras.json", name, primary_fw, api_path, details, catalogs)

    # Strategy 3: Static Injection
    dataloader_defaults = get_dataloader_semantics()
    for op, defn in dataloader_defaults.items():
      self.staged_specs["k_framework_extras.json"][op] = {
        "description": defn.get("description"),
        "std_args": defn.get("std_args"),
      }
      if "variants" in defn:
        for fw, variant in defn["variants"].items():
          if variant is not None:
            self.staged_mappings[fw][op] = variant

    # 4. Write to Disk
    if not semantics_path.exists():
      semantics_path.mkdir(parents=True, exist_ok=True)

    if not snapshots_path.exists():
      snapshots_path.mkdir(parents=True, exist_ok=True)

    # Write Hub (Specs)
    for filename, content in self.staged_specs.items():
      if content:
        self._write_json(semantics_path / filename, content, merge=True)

    # Write Spokes (Snapshots)
    for fw, mapping_data in self.staged_mappings.items():
      if mapping_data:
        try:
          if fw == "torch":
            import torch

            ver = torch.__version__
          else:
            ver = importlib.metadata.version(fw)
        except Exception:
          ver = "latest"

        file_data = {"__framework__": fw, "mappings": mapping_data}
        self._write_json(snapshots_path / f"{fw}_v{ver}_map.json", file_data, merge=True)

  def _get_ops_by_tier(self, tier: SemanticTier) -> Set[str]:
    if not hasattr(self.semantics, "_key_origins"):
      return set()
    return {k for k, v in self.semantics._key_origins.items() if v == tier.value}

  def _match_spec_op(self, api_name: str, spec_ops: Set[str]) -> Optional[str]:
    if api_name in spec_ops:
      return api_name
    lower_map = {k.lower(): k for k in spec_ops}
    if api_name.lower() in lower_map:
      return lower_map[api_name.lower()]
    matches = difflib.get_close_matches(api_name, spec_ops, n=1, cutoff=0.95)
    if matches:
      return matches[0]
    return None

  def _is_structurally_neural(self, api_path: str, kind: str) -> bool:
    heuristics = self._lazy_load_heuristics()
    patterns = heuristics.get("neural", [])
    for pat in patterns:
      if pat.search(api_path):
        return True
    if kind == "class" and ("Layer" in api_path or "Module" in api_path):
      return True
    return False

  def _is_structurally_extra(self, api_path: str, name: str) -> bool:
    heuristics = self._lazy_load_heuristics()
    patterns = heuristics.get("extras", [])
    for pat in patterns:
      if pat.search(api_path):
        return True
    name_lower = name.lower()
    if any(k in name_lower for k in ["seed", "save", "load", "device", "print", "info", "check"]):
      return True
    return False

  def _register_entry(
    self, target_filename: str, op_name: str, primary_fw: str, primary_path: str, details: Dict, catalogs: Dict[str, Dict]
  ):
    existing_def = self.semantics.data.get(op_name, {})

    spec_entry = {
      "description": existing_def.get("description", details.get("docstring_summary", "")),
      "type": details.get("type", "function"),
      "std_args": existing_def.get("std_args", details.get("params", [])),
    }

    self.staged_specs[target_filename][op_name] = spec_entry

    self.staged_mappings[primary_fw][op_name] = {"api": primary_path, "args": {p: p for p in details.get("params", [])}}

    primary_params = details.get("params", [])

    for other_fw, other_cat in catalogs.items():
      if other_fw == primary_fw:
        continue

      if primary_path in other_cat:
        self._register_mapping(op_name, other_fw, primary_path, other_cat[primary_path])
        continue

      fuzzy_match = self._find_fuzzy_match(other_cat, op_name, primary_params)
      if fuzzy_match:
        path, d = fuzzy_match
        self._register_mapping(op_name, other_fw, path, d)

  def _register_mapping(self, op_name: str, fw: str, path: str, details: Dict):
    self.staged_mappings[fw][op_name] = {"api": path, "args": {p: p for p in details.get("params", [])}}

  def _find_fuzzy_match(self, catalog: Dict, target_name: str, reference_params: List[str]) -> Optional[Tuple[str, Dict]]:
    best_score = 0.0
    best_match = None
    ref_arity = len(reference_params)

    for path, details in catalog.items():
      leaf_name = path.split(".")[-1]
      candidate_params = details.get("params", [])
      cand_arity = len(candidate_params)
      has_varargs = details.get("has_varargs", False)

      base_boost = 0.0
      if target_name.lower() == leaf_name.lower():
        base_boost = 0.3
      elif target_name in leaf_name or leaf_name in target_name:
        base_boost = 0.1

      name_ratio = difflib.SequenceMatcher(None, target_name.lower(), leaf_name.lower()).ratio()
      raw_score = name_ratio + base_boost

      arity_diff = abs(ref_arity - cand_arity)
      final_penalty = 0.0
      if arity_diff > 0:
        if has_varargs:
          final_penalty = 0.0
        elif arity_diff == 1:
          final_penalty = self.arity_penalty * 0.5
        else:
          final_penalty = self.arity_penalty

      final_score = raw_score - final_penalty

      if final_score >= 1.25 and final_penalty == 0:
        return path, details

      if final_score > best_score:
        best_score = final_score
        best_match = (path, details)

    if best_score >= self.similarity_threshold:
      return best_match
    return None

  def _write_json(self, path: Path, new_data: Dict, merge: bool = False):
    if merge and path.exists():
      try:
        with open(path, "rt", encoding="utf-8") as f:
          existing = json.load(f)
        if "mappings" in new_data and "mappings" in existing:
          existing["mappings"].update(new_data["mappings"])
          existing.update({k: v for k, v in new_data.items() if k != "mappings"})
          final_data = existing
        else:
          existing.update(new_data)
          final_data = existing
      except json.JSONDecodeError:
        final_data = new_data
    else:
      final_data = new_data

    with open(path, "wt", encoding="utf-8") as f:
      json.dump(final_data, f, indent=2, sort_keys=True)

    log_success(f"Updated [path]{path.name}[/path] ({len(new_data.get('mappings', new_data))} entries generated)")
