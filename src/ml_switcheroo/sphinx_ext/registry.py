"""
Registry scanning logic for the WASM demo.

This module is responsible for introspecting the installed Framework Adapters,
determining hierarchical relationships (Parent/Child frameworks), and
collecting tiered examples to be embedded in the static site.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.sphinx_ext.types import HierarchyMap


def scan_registry() -> Tuple[HierarchyMap, str, str]:
  """
  Scans registered adapters to build hierarchy, examples, and tier metadata.

  Returns:
      Tuple[HierarchyMap, str, str]:
          - The hierarchy dictionary mapping frameworks.
          - JSON string of preloaded examples.
          - JSON string of framework tier mapping.
  """
  fws = available_frameworks()
  priorities = get_framework_priority_order()

  # 1. Build Node Map
  hierarchy: HierarchyMap = defaultdict(list)
  tier_metadata: Dict[str, List[str]] = {}

  # Track roots explicitly
  roots = set()

  for key in fws:
    adapter = get_adapter(key)
    if not adapter:
      continue

    label = getattr(adapter, "display_name", key.capitalize())
    parent = getattr(adapter, "inherits_from", None)

    # Extract Tiers
    tiers = []
    if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:
      tiers = [t.value for t in adapter.supported_tiers]
    else:
      tiers = ["array", "neural", "extras"]
    tier_metadata[key] = tiers

    if parent:
      # This is a child node (e.g. flax_nnx -> jax)
      hierarchy[parent].append({"key": key, "label": label})
    else:
      # This is a root node (e.g. torch, jax, sass, rdna)
      roots.add(key)

  # 2. Convert to Render-Ready structures & Gather Examples
  examples = {}

  # Ensure we iterate roots sorted by priority provided by config
  sorted_roots = sorted(
    list(roots),
    key=lambda x: priorities.index(x) if x in priorities else 999,
  )

  # FIX: Iterate over sorted_roots to ensure keys exist in final_hierarchy
  final_hierarchy = {root: sorted(hierarchy.get(root, []), key=lambda x: x["label"]) for root in sorted_roots}

  # Collect Examples from Adapters
  for key in fws:
    adapter = get_adapter(key)
    if hasattr(adapter, "get_tiered_examples"):
      tiers = adapter.get_tiered_examples()
      parent_key = getattr(adapter, "inherits_from", None)

      for tier_name, code in tiers.items():
        uid = f"{key}_{tier_name}"

        if parent_key:
          src_fw = parent_key
          src_flavour = key
        else:
          src_fw = key
          src_flavour = None

        req_tier = "extras"
        if "math" in tier_name:
          req_tier = "array"
        elif "neural" in tier_name:
          req_tier = "neural"

        clean_tier_name = tier_name.replace("tier", "")
        clean_label = (
          clean_tier_name.split("_")[-1].capitalize() if "_" in clean_tier_name else clean_tier_name.capitalize()
        )

        display_fw = getattr(adapter, "display_name", key.title())
        label = f"{display_fw}: {clean_label}"

        # Dynamic Target Heuristic
        tgt_fw = None
        tgt_flavour = None

        candidates = [fw for fw in priorities if fw != src_fw and fw != parent_key and fw != src_flavour]

        if candidates:
          try:
            curr_idx = priorities.index(src_fw)
            rotated = priorities[curr_idx + 1 :] + priorities[:curr_idx]
            for c in rotated:
              if c in candidates:
                tgt_fw = c
                break
          except ValueError:
            tgt_fw = candidates[0]

        if not tgt_fw:
          tgt_fw = "target_placeholder"

        if tgt_fw in final_hierarchy and final_hierarchy[tgt_fw]:
          tgt_flavour = final_hierarchy[tgt_fw][0]["key"]

        examples[uid] = {
          "label": label,
          "srcFw": src_fw,
          "srcFlavour": src_flavour,
          "tgtFw": tgt_fw,
          "tgtFlavour": tgt_flavour,
          "code": code,
          "requiredTier": req_tier,
        }

  return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)
