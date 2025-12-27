"""
Scaffolding Tool for Knowledge Base Discovery.

This module provides the `Scaffolder` class, which automates the population of
the Semantic Knowledge Base by scanning installed libraries (e.g., Torch, JAX).
It uses heuristic pattern matching to categorize discovered APIs into semantic
tiers (Neural vs Math vs Extras) and persists them into the `semantics/` JSON
specifications and `snapshots/` mapping overlays.

Optimization Update:
- Implements Indexing for Catalog Lookups to prevent O(N^2) complexity.
- Uses pre-computed indices keyed by lowercase API names for fast retrieval.
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


class Scaffolder:
  """
  Automated discovery tool that aligns framework APIs.

  This class scans multiple frameworks, identifies common operations based
  on name similarity (e.g., 'torch.abs' == 'jax.numpy.abs'), and generates
  the initial JSON mappings required for the transpiler.
  """

  def __init__(
    self,
    semantics: Optional[SemanticsManager] = None,
    similarity_threshold: float = 0.8,
    arity_penalty: float = 0.3,
  ):
    """
    Initializes the scaffolder.

    Args:
        semantics: Existing knowledge base to check against (optional).
        similarity_threshold: Levenshtein distance threshold (0.0 - 1.0) for fuzzy matching.
        arity_penalty: Score penalty for operations with differing argument counts.
    """
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

    # Optimization: Lookup Indices for O(1) matching
    self._catalog_indices: Dict[str, Dict[str, List[Tuple[str, Dict]]]] = {}

  def _lazy_load_heuristics(self) -> Dict[str, List[re.Pattern]]:
    """
    Loads regex heuristics from all registered framework adapters.

    Returns:
        A dictionary mapping Sematic Tier names to lists of compiled regex patterns.
    """
    if self._cached_heuristics is not None:
      return self._cached_heuristics

    compiled: Dict[str, List[re.Pattern]] = {"neural": [], "extras": []}
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

  def _build_catalog_index(self, fw: str, catalog: Dict[str, Any]):
    """
    Creates a fast lookup index for a framework catalog.

    Args:
        fw: Framework key.
        catalog: The raw catalog dictionary from the Inspector.
    """
    index = defaultdict(list)
    for path, details in catalog.items():
      name = details["name"].lower()
      index[name].append((path, details))
    self._catalog_indices[fw] = index

  def scaffold(self, frameworks: List[str], root_dir: Optional[Path] = None):
    """
    Main entry point. Scans frameworks and builds/updates JSON mappings.

    1. Scans all requested frameworks using `ApiInspector`.
    2. Aligns APIs against known standards (Specs).
    3. Uses fuzzy matching to align APIs between frameworks.
    4. Writes results to disk (semantics/ and snapshots/).

    Args:
        frameworks: List of framework keys to scan (e.g. `['torch', 'jax']`).
        root_dir: Optional root directory path. Defaults to package paths.
    """
    if root_dir:
      semantics_path = root_dir / "semantics"
      snapshots_path = root_dir / "snapshots"
    else:
      semantics_path = resolve_semantics_dir()
      snapshots_path = resolve_snapshots_dir()

    # Pre-load known specs
    known_neural_ops = self._get_ops_by_tier(SemanticTier.NEURAL)
    known_math_ops = self._get_ops_by_tier(SemanticTier.ARRAY_API)
    known_extras_ops = self._get_ops_by_tier(SemanticTier.EXTRAS)

    catalogs = {}

    # --- PHASE 1: SCANNING ---
    for fw in frameworks:
      log_info(f"Scanning [code]{fw}[/code]...")

      adapter = get_adapter(fw)

      # 1. Determine Search Paths
      scan_targets = [fw]
      if adapter and hasattr(adapter, "search_modules") and adapter.search_modules:
        scan_targets = adapter.search_modules

      # 2. Determine Unsafe Modules from Adapter
      unsafe_blacklist = set()
      if adapter:
        # Safely get property, defaulting to empty if adapter is older version (partial implementation)
        unsafe_blacklist = getattr(adapter, "unsafe_submodules", set())

      fw_catalog = {}
      for module_name in scan_targets:
        try:
          # Pass blacklist to inspector
          module_catalog = self.inspector.inspect(module_name, unsafe_modules=unsafe_blacklist)
          fw_catalog.update(module_catalog)
        except Exception as e:
          log_info(f"Skipping module {module_name}: {e}")

      catalogs[fw] = fw_catalog
      self._build_catalog_index(fw, catalogs[fw])

    primary_fw = "torch" if "torch" in frameworks else frameworks[0]

    primary_items = []
    if primary_fw in catalogs and catalogs[primary_fw]:
      primary_items = list(catalogs[primary_fw].items())
      log_info(f"Aligning APIs against [code]{primary_fw}[/code] and [code]Specs[/code]...")
    else:
      log_info(f"Primary framework {primary_fw} scan empty or failed. Skipping heuristic alignment.")

    for api_path, details in primary_items:
      name = details["name"]
      kind = details.get("type", "function")

      # Strategy 1: Spec Validation
      neural_match = self._match_spec_op(name, known_neural_ops)
      if neural_match:
        self._register_entry(
          "k_neural_net.json",
          neural_match,
          primary_fw,
          api_path,
          details,
          catalogs,
        )
        continue

      math_match = self._match_spec_op(name, known_math_ops)
      if math_match:
        self._register_entry(
          "k_array_api.json",
          math_match,
          primary_fw,
          api_path,
          details,
          catalogs,
        )
        continue

      extras_match = self._match_spec_op(name, known_extras_ops)
      if extras_match:
        self._register_entry(
          "k_framework_extras.json",
          extras_match,
          primary_fw,
          api_path,
          details,
          catalogs,
        )
        continue

      # Strategy 2: Heuristic Fallback
      if self._is_structurally_neural(api_path, kind):
        self._register_entry("k_neural_net.json", name, primary_fw, api_path, details, catalogs)
        continue

      if self._is_structurally_extra(api_path, name):
        self._register_entry(
          "k_framework_extras.json",
          name,
          primary_fw,
          api_path,
          details,
          catalogs,
        )
        continue

      self._register_entry("k_framework_extras.json", name, primary_fw, api_path, details, catalogs)

    # 4. Write to Disk
    if not semantics_path.exists():
      semantics_path.mkdir(parents=True, exist_ok=True)

    if not snapshots_path.exists():
      snapshots_path.mkdir(parents=True, exist_ok=True)

    for filename, content in self.staged_specs.items():
      if content:
        self._write_json(semantics_path / filename, content, merge=True)

    for fw, mapping_data in self.staged_mappings.items():
      if mapping_data:
        try:
          ver_name = fw
          if fw == "flax_nnx":
            ver_name = "flax"
          ver = importlib.metadata.version(ver_name)
        except Exception:
          ver = "latest"

        file_data = {"__framework__": fw, "mappings": mapping_data}
        self._write_json(snapshots_path / f"{fw}_v{ver}_map.json", file_data, merge=True)

  def _get_ops_by_tier(self, tier: SemanticTier) -> Set[str]:
    """
    Retrieves known operations belonging to a specific semantic tier.
    """
    if not hasattr(self.semantics, "_key_origins"):
      return set()
    return {k for k, v in self.semantics._key_origins.items() if v == tier.value}

  def _match_spec_op(self, api_name: str, spec_ops: Set[str]) -> Optional[str]:
    """
    Checks if an API name exists in a set of known spec operations using fuzzy matching.
    """
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
    """
    Determines if an API looks like a Neural Network layer/component via regex heuristics.
    """
    heuristics = self._lazy_load_heuristics()
    patterns = heuristics.get("neural", [])
    for pat in patterns:
      if pat.search(api_path):
        return True
    if kind == "class" and ("Layer" in api_path or "Module" in api_path):
      return True
    return False

  def _is_structurally_extra(self, api_path: str, name: str) -> bool:
    """
    Determines if an API looks like a Utility/Extra via regex heuristics.
    """
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
    self,
    target_filename: str,
    op_name: str,
    primary_fw: str,
    primary_path: str,
    details: Dict,
    catalogs: Dict[str, Dict],
  ):
    """
    Registers a found operation into the Staging Areas.
    Attempts to find matching variants in other frameworks.
    """
    existing_def = self.semantics.data.get(op_name, {})

    spec_entry = {
      "description": existing_def.get("description", details.get("docstring_summary", "")),
      "type": details.get("type", "function"),
      "std_args": existing_def.get("std_args", details.get("params", [])),
    }

    self.staged_specs[target_filename][op_name] = spec_entry
    self.staged_mappings[primary_fw][op_name] = {
      "api": primary_path,
      "args": {p: p for p in details.get("params", [])},
    }

    primary_params = details.get("params", [])

    for other_fw, other_cat in catalogs.items():
      if other_fw == primary_fw:
        continue

      if primary_path in other_cat:
        self._register_mapping(op_name, other_fw, primary_path, other_cat[primary_path])
        continue

      fuzzy_match = self._find_fuzzy_match(other_cat, op_name, primary_params, fw_key=other_fw)
      if fuzzy_match:
        path, d = fuzzy_match
        self._register_mapping(op_name, other_fw, path, d)

  def _register_mapping(self, op_name: str, fw: str, path: str, details: Dict):
    """Helper to stage a mapping."""
    self.staged_mappings[fw][op_name] = {
      "api": path,
      "args": {p: p for p in details.get("params", [])},
    }

  def _find_fuzzy_match(
    self,
    catalog: Dict,
    target_name: str,
    reference_params: List[str],
    fw_key: str = None,
  ) -> Optional[Tuple[str, Dict]]:
    """
    Finds the closest matching API in a catalog to the target name.
    Uses indexed lookup first, then fuzzy scoring with arity penalties.
    """
    target_lower = target_name.lower()
    candidates = []
    using_fuzzy_score = False

    if fw_key and fw_key in self._catalog_indices:
      if target_lower in self._catalog_indices[fw_key]:
        candidates.extend(self._catalog_indices[fw_key][target_lower])

    if not candidates and len(catalog) < 2000:
      using_fuzzy_score = True

      source_names = []
      if fw_key and fw_key in self._catalog_indices:
        source_names = list(self._catalog_indices[fw_key].keys())
      else:
        source_names = [d["name"].lower() for d in catalog.values()]

      loose_cutoff = min(0.4, self.similarity_threshold)
      matches = difflib.get_close_matches(target_lower, source_names, n=5, cutoff=loose_cutoff)

      if fw_key and fw_key in self._catalog_indices:
        for m in matches:
          candidates.extend(self._catalog_indices[fw_key][m])
      else:
        for path, details in catalog.items():
          if details["name"].lower() in matches:
            candidates.append((path, details))

    if not candidates:
      return None

    best_score = 0.0
    best_match = None
    ref_arity = len(reference_params)

    for path, details in candidates:
      leaf_name = details["name"]
      candidate_params = details.get("params", [])
      cand_arity = len(candidate_params)
      has_varargs = details.get("has_varargs", False)

      if using_fuzzy_score:
        raw_score = difflib.SequenceMatcher(None, target_lower, leaf_name.lower()).ratio()
        l_low = leaf_name.lower()
        if len(target_lower) >= 3 and len(l_low) >= 3:
          if target_lower.startswith(l_low) or l_low.startswith(target_lower):
            raw_score = max(raw_score, 0.85)

      else:
        raw_score = 1.0

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

      if final_score >= 1.0 and final_penalty == 0:
        return path, details

      if final_score > best_score:
        best_score = final_score
        best_match = (path, details)

    if best_score >= self.similarity_threshold:
      return best_match
    return None

  def _write_json(self, path: Path, new_data: Dict, merge: bool = False):
    """
    Persists JSON data to disk, merging with existing files if requested.
    """
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

    log_success(f"Updated [path]{path.name}[/path]")
