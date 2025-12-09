"""
Scaffolding Tool for Knowledge Base Discovery.

This module provides the `Scaffolder` class, which inspects installed
libraries (Torch, JAX, etc.) and aligns them against the Specification-Guided
Knowledge Base.

It prioritizes matching discovered APIs against ingested Specs (ONNX, Array API)
before falling back to structural heuristics provided by Framework Adapters.
"""

import json
import difflib
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.utils.console import console, log_info, log_success
from ml_switcheroo.frameworks import available_frameworks, get_adapter


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
    """
    Initializes the Scaffolder.

    Args:
        semantics: Pre-loaded SemanticsManager (optional).
        similarity_threshold: Min ratio to consider a fuzzy match (default 0.8).
        arity_penalty: Score reduction if parameter counts mismatch (default 0.3).
    """
    self.inspector = ApiInspector()
    self.console = console
    self.semantics = semantics or SemanticsManager()
    self.similarity_threshold = similarity_threshold
    self.arity_penalty = arity_penalty

    # Staging areas for the 3 Tiers
    self.tier_a_math: Dict[str, Any] = {}  # Array API
    self.tier_b_neural: Dict[str, Any] = {}  # Neural Net (ONNX)
    self.tier_c_extras: Dict[str, Any] = {}  # Framework Extras

    self._cached_heuristics: Optional[Dict[str, List[re.Pattern]]] = None

  def _lazy_load_heuristics(self) -> Dict[str, List[re.Pattern]]:
    """
    Aggregates discovery regexes from all registered framework adapters.
    Uses result caching for performance.
    """
    if self._cached_heuristics is not None:
      return self._cached_heuristics

    compiled: Dict[str, List[re.Pattern]] = {"neural": [], "extras": []}

    # Loop through every registered framework and pull its rules
    for fw_name in available_frameworks():
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      # Check if adapter supports the new property (duck typing for backward compat)
      if hasattr(adapter, "discovery_heuristics") and adapter.discovery_heuristics:
        heuristics = adapter.discovery_heuristics

        for tier, patterns in heuristics.items():
          if tier in compiled:
            for pat in patterns:
              try:
                # Case insensitive matching allows matching "Linear" or "linear"
                compiled[tier].append(re.compile(pat, re.IGNORECASE))
              except re.error as e:
                print(f"⚠️ Invalid regex pattern in {fw_name} adapter: '{pat}' - {e}")

    self._cached_heuristics = compiled
    return compiled

  def scaffold(self, frameworks: List[str], output_dir: Path):
    """
    Main entry point. Scans frameworks and builds/updates JSON mappings.

    Discovery Strategy:
    1. **Spec Validation**: Checks if discovered API matches a specific
       Abstract Operator already defined in the Knowledge Base (e.g., from ONNX).
    2. **Heuristic Fallback**: If no spec match, checks Framework Adapters
       for regex conventions (e.g. `.nn` -> Neural).

    Args:
        frameworks: List of package names (e.g. ['torch', 'jax']).
        output_dir: Directory to save generated JSON files.
    """
    # Pre-load known specs to drive categorization
    known_neural_ops = self._get_ops_by_tier(SemanticTier.NEURAL)
    known_math_ops = self._get_ops_by_tier(SemanticTier.ARRAY_API)
    # Tier C Knowledge (if present in semantics)
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

      # --- Strategy 1: Spec Validation (Priority) ---
      # Check if this name matches a known Abstract Op from the specs
      neural_match = self._match_spec_op(name, known_neural_ops)
      if neural_match:
        self._update_entry(self.tier_b_neural, neural_match, primary_fw, api_path, details, catalogs)
        continue

      math_match = self._match_spec_op(name, known_math_ops)
      if math_match:
        self._update_entry(self.tier_a_math, math_match, primary_fw, api_path, details, catalogs)
        continue

      extras_match = self._match_spec_op(name, known_extras_ops)
      if extras_match:
        self._update_entry(self.tier_c_extras, extras_match, primary_fw, api_path, details, catalogs)
        continue

      # --- Strategy 2: Heuristic Fallback (Dynamic) ---
      # If not in specs, we guess based on structure using aggregated regex rules.

      if self._is_structurally_neural(api_path, kind):
        self._update_entry(self.tier_b_neural, name, primary_fw, api_path, details, catalogs)
        continue

      if self._is_structurally_extra(api_path, name):
        self._update_entry(self.tier_c_extras, name, primary_fw, api_path, details, catalogs)
        continue

      # Heuristic: Math Tier (Default)
      self._update_entry(self.tier_a_math, name, primary_fw, api_path, details, catalogs)

    if not output_dir.exists():
      output_dir.mkdir(parents=True)

    self._write_json(output_dir / "k_array_api.json", self.tier_a_math)
    self._write_json(output_dir / "k_neural_net.json", self.tier_b_neural)
    self._write_json(output_dir / "k_framework_extras.json", self.tier_c_extras)

  def _get_ops_by_tier(self, tier: SemanticTier) -> Set[str]:
    """Retrieves listing of known abstract operations for a given tier."""
    # Access internal storage of sources to determine tier.
    # Fallback to empty if using a manager that hasn't indexed origins.
    if not hasattr(self.semantics, "_key_origins"):
      return set()

    return {k for k, v in self.semantics._key_origins.items() if v == tier.value}

  def _match_spec_op(self, api_name: str, spec_ops: Set[str]) -> Optional[str]:
    """
    Attempts to find a matching Abstract Op ID in the spec set.
    Performs exact match, case-insensitive match, and fuzzy match.
    """
    # 1. Exact Match
    if api_name in spec_ops:
      return api_name

    # 2. Case Insensitive (e.g. ReLU vs Relu)
    lower_map = {k.lower(): k for k in spec_ops}
    if api_name.lower() in lower_map:
      return lower_map[api_name.lower()]

    # 3. Fuzzy Match (High Strictness for Spec alignment)
    matches = difflib.get_close_matches(api_name, spec_ops, n=1, cutoff=0.95)
    if matches:
      return matches[0]

    return None

  def _is_structurally_neural(self, api_path: str, kind: str) -> bool:
    """
    Determines if API is likely Neural based on regex patterns registered via adapters.
    """
    heuristics = self._lazy_load_heuristics()
    patterns = heuristics.get("neural", [])

    # Check RegEx matches against the full path
    for pat in patterns:
      if pat.search(api_path):
        return True

    # Core Fallback for Classes containing 'Layer' or 'Module' even if no adapter
    # explicitly claimed them (safety net for bootstrapping)
    if kind == "class" and ("Layer" in api_path or "Module" in api_path):
      return True

    return False

  def _is_structurally_extra(self, api_path: str, name: str) -> bool:
    """
    Determines if API is likely a Framework Utility (Tier C) based on patterns.
    """
    heuristics = self._lazy_load_heuristics()
    patterns = heuristics.get("extras", [])

    # 1. Check Regex
    for pat in patterns:
      if pat.search(api_path):
        return True

    # 2. Fallbacks for unregistered frameworks (Bootstrapping)
    name_lower = name.lower()
    # Common utility keywords
    if any(k in name_lower for k in ["seed", "save", "load", "device", "print", "info", "check"]):
      return True

    return False

  def _update_entry(
    self,
    target_dict: Dict,
    op_name: str,
    primary_fw: str,
    primary_path: str,
    details: Dict,
    catalogs: Dict[str, Dict],
  ):
    """
    Creates or updates the mapping entry in the staging dict.
    """
    # If the entry exists (maybe loaded from spec), preserve description/args
    # If new, create skeleton.
    if op_name not in target_dict:
      # We check if SemanticsManager already has data for this op (from Spec)
      existing_def = self.semantics.data.get(op_name, {})

      target_dict[op_name] = {
        "description": existing_def.get("description", details.get("docstring_summary", "")),
        "type": details.get("type", "function"),
        "std_args": existing_def.get("std_args", details.get("params", [])),
        "variants": existing_def.get("variants", {}),
      }

    # Update Primary Variant
    target_dict[op_name]["variants"][primary_fw] = {
      "api": primary_path,
      "args": {p: p for p in details.get("params", [])},
    }

    # Primary Params context for fuzzy matching
    primary_params = details.get("params", [])

    # Hunt for variants in other frameworks
    for other_fw, other_cat in catalogs.items():
      if other_fw == primary_fw:
        continue

      # 1. Exact Path Match (Rare across frameworks)
      if primary_path in other_cat:
        self._register_match(target_dict[op_name], other_fw, primary_path, other_cat[primary_path])
        continue

      # 2. Match against Abstract Name (op_name) with Signature Awareness
      fuzzy_match = self._find_fuzzy_match(other_cat, op_name, primary_params)
      if fuzzy_match:
        path, d = fuzzy_match
        self._register_match(target_dict[op_name], other_fw, path, d)

  def _register_match(self, entry: Dict, fw: str, path: str, details: Dict):
    """Helper to write the variant dict."""
    entry["variants"][fw] = {
      "api": path,
      "args": {p: p for p in details.get("params", [])},
    }

  def _find_fuzzy_match(self, catalog: Dict, target_name: str, reference_params: List[str]) -> Optional[Tuple[str, Dict]]:
    """
    Finds the best API match using name similarity AND signature compatibility.
    """
    best_score = 0.0
    best_match = None

    ref_arity = len(reference_params)

    for path, details in catalog.items():
      leaf_name = path.split(".")[-1]
      candidate_params = details.get("params", [])
      cand_arity = len(candidate_params)

      # Feature 07: Check for varargs
      has_varargs = details.get("has_varargs", False)

      # --- 1. Calculate Name Similarity ---
      # Boost exact substring matches
      base_boost = 0.0
      if target_name.lower() == leaf_name.lower():
        base_boost = 0.3
      elif target_name in leaf_name or leaf_name in target_name:
        base_boost = 0.1

      name_ratio = difflib.SequenceMatcher(None, target_name.lower(), leaf_name.lower()).ratio()

      raw_score = name_ratio + base_boost

      # --- 2. Sanity Check: Arity (Signature Analysis) ---
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

  def _write_json(self, path: Path, new_data: Dict):
    """Writes data to JSON, merging with existing on disk if present."""
    if path.exists():
      try:
        with open(path, "rt", encoding="utf-8") as f:
          existing = json.load(f)
        existing.update(new_data)
        final_data = existing
      except json.JSONDecodeError:
        final_data = new_data
    else:
      final_data = new_data

    with open(path, "wt", encoding="utf-8") as f:
      json.dump(final_data, f, indent=2, sort_keys=True)

    log_success(f"Written {len(final_data)} entries to [path]{path}[/path]")
