"""Registry scanning logic for the WASM demo.

This module is responsible for introspecting the installed Framework Adapters,
determining hierarchical relationships (Parent/Child frameworks), and
collecting tiered examples to be embedded in the static site.
"""

import json
import typing
from collections import defaultdict

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.sphinx_ext.types import HierarchyMap

if typing.TYPE_CHECKING:
  from ml_switcheroo.frameworks.base import FrameworkAdapter


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


def _extract_adapter_tiers(adapter: "FrameworkAdapter") -> list[str]:
  """Return supported semantic tier names for ``adapter``, falling back to all standard tiers.

  Args:
      adapter: Registered framework adapter instance.

  Returns:
      List of supported tier identifier strings.
  """
  try:
    supp_tiers = adapter.supported_tiers
  except AttributeError:
    supp_tiers = []
  return [t.value for t in supp_tiers] if supp_tiers else ["array", "neural", "extras"]


def _resolve_default_target(
  src_fw: str,
  parent_key: str | None,
  src_flavour: str | None,
  priorities: list[str],
  final_hierarchy: HierarchyMap,
) -> tuple[str, str | None]:
  """Select the default target framework and optional flavour for an example starting from ``src_fw``.

  Args:
      src_fw: Effective source root framework key.
      parent_key: Optional parent framework key when ``adapter`` is a child flavour.
      src_flavour: Optional source flavour key.
      priorities: Ordered framework priority list.
      final_hierarchy: Sorted parent-to-children framework hierarchy.

  Returns:
      Tuple of ``(target_framework_key, optional_target_flavour_key)``.
  """
  tgt_fw: str | None = None
  candidates = [fw for fw in priorities if fw not in (src_fw, parent_key, src_flavour)]
  if candidates:
    try:
      curr_idx = priorities.index(src_fw)
      rotated = priorities[curr_idx + 1 :] + priorities[:curr_idx]
      tgt_fw = next((c for c in rotated if c in candidates), None)
    except ValueError:
      tgt_fw = candidates[0]

  resolved_tgt = tgt_fw or "target_placeholder"
  tgt_flavour = final_hierarchy[resolved_tgt][0]["key"] if final_hierarchy.get(resolved_tgt) else None
  return resolved_tgt, tgt_flavour


def _collect_adapter_examples(
  key: str,
  adapter: "FrameworkAdapter",
  priorities: list[str],
  final_hierarchy: HierarchyMap,
) -> dict[str, dict[str, str | bool | None]]:
  """Build interactive demo example records from ``adapter.get_tiered_examples()``.

  Args:
      key: Registered adapter key.
      adapter: Registered framework adapter instance.
      priorities: Ordered framework priority list.
      final_hierarchy: Sorted parent-to-children framework hierarchy.

  Returns:
      Dictionary of example entries keyed by ``<key>_<tier_name>``.
  """
  collected: dict[str, dict[str, str | bool | None]] = {}
  parent_key = adapter.inherits_from
  src_fw = parent_key if parent_key else key
  src_flavour = key if parent_key else None
  display_fw = str(adapter.display_name) if adapter.display_name else key.title()
  tgt_fw, tgt_flavour = _resolve_default_target(src_fw, parent_key, src_flavour, priorities, final_hierarchy)

  for tier_name, code in adapter.get_tiered_examples().items():
    req_tier = "array" if "math" in tier_name else ("neural" if "neural" in tier_name else "extras")
    clean_tier_name = tier_name.replace("tier", "")
    clean_label = clean_tier_name.split("_")[-1].capitalize() if "_" in clean_tier_name else clean_tier_name.capitalize()
    collected[f"{key}_{tier_name}"] = {
      "label": f"{display_fw}: {clean_label}",
      "srcFw": src_fw,
      "srcFlavour": src_flavour,
      "tgtFw": tgt_fw,
      "tgtFlavour": tgt_flavour,
      "code": code,
      "requiredTier": req_tier,
    }
  return collected


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

  hierarchy: HierarchyMap = defaultdict(list)
  tier_metadata: dict[str, list[str]] = {}
  roots: set[str] = set()

  for key in fws:
    adapter = get_adapter(key)
    if not adapter:
      continue
    label = str(adapter.display_name) if adapter.display_name else key.capitalize()
    tier_metadata[key] = _extract_adapter_tiers(adapter)
    if adapter.inherits_from:
      hierarchy[adapter.inherits_from].append({"key": key, "label": label})
    else:
      roots.add(key)

  sorted_roots = sorted(roots, key=lambda x: priorities.index(x) if x in priorities else 999)
  final_hierarchy: HierarchyMap = {
    root: sorted(hierarchy.get(root, []), key=lambda x: x["label"]) for root in sorted_roots
  }

  examples: dict[str, dict[str, str | bool | None]] = {}
  for key in fws:
    adapter = get_adapter(key)
    if adapter is not None:
      examples.update(_collect_adapter_examples(key, adapter, priorities, final_hierarchy))

  examples.update(
    {
      preset_uid: preset_data
      for preset_uid, preset_data in _build_cross_dialect_dynamic_presets().items()
      if preset_data["srcFw"] in fws and preset_data["tgtFw"] in fws
    }
  )

  return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)
