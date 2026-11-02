"""Registry scanning logic for the WASM demo.

This module is responsible for introspecting the installed Framework Adapters,
determining hierarchical relationships (Parent/Child frameworks), and
collecting tiered examples to be embedded in the static site.
"""

import json
from collections import defaultdict

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.sphinx_ext.types import HierarchyMap


def _build_cross_dialect_dynamic_presets() -> dict[str, dict[str, str | bool | None]]:
  """Build interactive browser presets demonstrating input-dependent shape interweaving across dialects.

  Returns:
      Dictionary of preset example records keyed by unique example ID.
  """
  dyn_torch_code = """import torch
import torch.nn as nn

class DynamicShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten dynamic spatial dimensions (2, 3, 4, 4) -> (2, 48)
        self.flat = nn.Flatten()
        # LazyLinear resolves in_features=48 on-device
        self.fc = nn.LazyLinear(10)

    def forward(self, x):
        # Dynamic reshape with -1 and symbolic batch size
        x = self.flat(x)  # flatten spatial dims
        x = x.view(x.size(0), -1)  # symbolic -1 view
        return self.fc(x)
"""
  rdna_code = """; RDNA Code Generation Initialized (Arch: gfx1030)
; Input x -> v0
; Custom RDNA GEMM kernel lifted to high-level framework
; BEGIN Linear (fc1)
v_mov_b32 v1, 0
s_mov_b32 s0, 0
L_GEMM_fc1:
global_load_dword v2, [v4], off
global_load_dword v3, [v5], off
s_waitcnt vmcnt(0)
v_fmac_f32 v1, v2, v3
v_add_u32 v4, v4, 4
v_add_u32 v5, v5, 4
s_add_i32 s0, s0, 1
s_cmp_lt_i32 s0, 16
s_cbranch_scc1 L_GEMM_fc1
; END Linear (fc1)
; Return: v1
"""
  shlo_code = """module {
  func.func @forward(%x: tensor<2x16xf32>) -> tensor<2x16xf32> {
    // StableHLO elementwise + custom_call lifted on-device
    %0 = stablehlo.abs %x : tensor<2x16xf32>
    %1 = stablehlo.custom_call @relu(%0) {call_target_name = "@relu"} : (tensor<2x16xf32>) -> tensor<2x16xf32>
    func.return %1 : tensor<2x16xf32>
  }
}
"""
  return {
    "preset_dyn_torch_to_flax_nnx": {
      "label": "On-Device Dynamic: PyTorch (LazyLinear + view -1) -> Flax NNX",
      "srcFw": "torch",
      "srcFlavour": None,
      "tgtFw": "jax",
      "tgtFlavour": "flax_nnx",
      "code": dyn_torch_code,
      "requiredTier": "neural",
      "enableOnDevice": True,
      "inputShape": "(2, 3, 4, 4)",
    },
    "preset_dyn_torch_to_rdna": {
      "label": "On-Device Dynamic: PyTorch (Flatten + LazyLinear) -> RDNA ISA",
      "srcFw": "torch",
      "srcFlavour": None,
      "tgtFw": "rdna",
      "tgtFlavour": None,
      "code": dyn_torch_code,
      "requiredTier": "neural",
      "enableOnDevice": True,
      "inputShape": "(2, 3, 4, 4)",
    },
    "preset_dyn_torch_to_nvidia_sass": {
      "label": "On-Device Dynamic: PyTorch (Flatten + LazyLinear) -> NVIDIA SASS",
      "srcFw": "torch",
      "srcFlavour": None,
      "tgtFw": "nvidia_sass",
      "tgtFlavour": None,
      "code": dyn_torch_code,
      "requiredTier": "neural",
      "enableOnDevice": True,
      "inputShape": "(2, 3, 4, 4)",
    },
    "preset_dyn_torch_to_stablehlo": {
      "label": "On-Device Dynamic: PyTorch (Flatten + LazyLinear) -> StableHLO",
      "srcFw": "torch",
      "srcFlavour": None,
      "tgtFw": "stablehlo",
      "tgtFlavour": None,
      "code": dyn_torch_code,
      "requiredTier": "neural",
      "enableOnDevice": True,
      "inputShape": "(2, 3, 4, 4)",
    },
    "preset_dyn_rdna_to_flax_nnx": {
      "label": "On-Device Lift: RDNA ISA -> Flax NNX (Shape Recovery)",
      "srcFw": "rdna",
      "srcFlavour": None,
      "tgtFw": "jax",
      "tgtFlavour": "flax_nnx",
      "code": rdna_code,
      "requiredTier": "extras",
      "enableOnDevice": True,
      "inputShape": "(2, 16)",
    },
    "preset_dyn_stablehlo_to_torch": {
      "label": "On-Device Lift: StableHLO -> PyTorch (Shape Recovery)",
      "srcFw": "stablehlo",
      "srcFlavour": None,
      "tgtFw": "torch",
      "tgtFlavour": None,
      "code": shlo_code,
      "requiredTier": "extras",
      "enableOnDevice": True,
      "inputShape": "(2, 16)",
    },
  }


def scan_registry() -> tuple[HierarchyMap, str, str]:
  """Scan registered adapters to build hierarchy, examples, and tier metadata.

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
  tier_metadata: dict[str, list[str]] = {}

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
      # This is a root node (e.g. torch, jax, nvidia_sass, rdna)
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
      tiered_examples: dict[str, str] = getattr(adapter, "get_tiered_examples")()
      parent_key = getattr(adapter, "inherits_from", None)

      for tier_name, code in tiered_examples.items():
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

  for preset_uid, preset_data in _build_cross_dialect_dynamic_presets().items():
    if preset_data["srcFw"] in fws and preset_data["tgtFw"] in fws:
      examples[preset_uid] = preset_data

  return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)
