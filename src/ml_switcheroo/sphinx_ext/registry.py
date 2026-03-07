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
  fws = available_frameworks()  # pragma: no cover
  priorities = get_framework_priority_order()  # pragma: no cover

  # 1. Build Node Map
  hierarchy: HierarchyMap = defaultdict(list)  # pragma: no cover
  tier_metadata: Dict[str, List[str]] = {}  # pragma: no cover

  # Track roots explicitly
  roots = set()  # pragma: no cover

  for key in fws:  # pragma: no cover
    adapter = get_adapter(key)  # pragma: no cover
    if not adapter:  # pragma: no cover
      continue  # pragma: no cover

    label = getattr(adapter, "display_name", key.capitalize())  # pragma: no cover
    parent = getattr(adapter, "inherits_from", None)  # pragma: no cover

    # Extract Tiers
    tiers = []  # pragma: no cover
    if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:  # pragma: no cover
      tiers = [t.value for t in adapter.supported_tiers]  # pragma: no cover
    else:
      tiers = ["array", "neural", "extras"]  # pragma: no cover
    tier_metadata[key] = tiers  # pragma: no cover

    if parent:  # pragma: no cover
      # This is a child node (e.g. flax_nnx -> jax)
      hierarchy[parent].append({"key": key, "label": label})  # pragma: no cover
    else:
      # This is a root node (e.g. torch, jax, sass, rdna)
      roots.add(key)  # pragma: no cover

  # 2. Convert to Render-Ready structures & Gather Examples
  examples = {}  # pragma: no cover

  # Ensure we iterate roots sorted by priority provided by config
  sorted_roots = sorted(  # pragma: no cover
    list(roots),
    key=lambda x: priorities.index(x) if x in priorities else 999,
  )

  # FIX: Iterate over sorted_roots to ensure keys exist in final_hierarchy
  final_hierarchy = {
    root: sorted(hierarchy.get(root, []), key=lambda x: x["label"]) for root in sorted_roots
  }  # pragma: no cover

  # Collect Examples from Adapters
  for key in fws:  # pragma: no cover
    adapter = get_adapter(key)  # pragma: no cover
    if hasattr(adapter, "get_tiered_examples"):  # pragma: no cover
      tiers = adapter.get_tiered_examples()  # pragma: no cover
      parent_key = getattr(adapter, "inherits_from", None)  # pragma: no cover

      for tier_name, code in tiers.items():  # pragma: no cover
        uid = f"{key}_{tier_name}"  # pragma: no cover

        if parent_key:  # pragma: no cover
          src_fw = parent_key  # pragma: no cover
          src_flavour = key  # pragma: no cover
        else:
          src_fw = key  # pragma: no cover
          src_flavour = None  # pragma: no cover

        req_tier = "extras"  # pragma: no cover
        if "math" in tier_name:  # pragma: no cover
          req_tier = "array"  # pragma: no cover
        elif "neural" in tier_name:  # pragma: no cover
          req_tier = "neural"  # pragma: no cover

        clean_tier_name = tier_name.replace("tier", "")  # pragma: no cover
        clean_label = (  # pragma: no cover
          clean_tier_name.split("_")[-1].capitalize() if "_" in clean_tier_name else clean_tier_name.capitalize()
        )

        display_fw = getattr(adapter, "display_name", key.title())  # pragma: no cover
        label = f"{display_fw}: {clean_label}"  # pragma: no cover

        # Dynamic Target Heuristic
        tgt_fw = None  # pragma: no cover
        tgt_flavour = None  # pragma: no cover

        candidates = [
          fw for fw in priorities if fw != src_fw and fw != parent_key and fw != src_flavour
        ]  # pragma: no cover

        if candidates:  # pragma: no cover
          try:  # pragma: no cover
            curr_idx = priorities.index(src_fw)  # pragma: no cover
            rotated = priorities[curr_idx + 1 :] + priorities[:curr_idx]  # pragma: no cover
            for c in rotated:  # pragma: no cover
              if c in candidates:  # pragma: no cover
                tgt_fw = c  # pragma: no cover
                break  # pragma: no cover
          except ValueError:  # pragma: no cover
            tgt_fw = candidates[0]  # pragma: no cover

        if not tgt_fw:  # pragma: no cover
          tgt_fw = "target_placeholder"  # pragma: no cover

        if tgt_fw in final_hierarchy and final_hierarchy[tgt_fw]:  # pragma: no cover
          tgt_flavour = final_hierarchy[tgt_fw][0]["key"]  # pragma: no cover

        examples[uid] = {  # pragma: no cover
          "label": label,
          "srcFw": src_fw,
          "srcFlavour": src_flavour,
          "tgtFw": tgt_fw,
          "tgtFlavour": tgt_flavour,
          "code": code,
          "requiredTier": req_tier,
        }

  return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)  # pragma: no cover
