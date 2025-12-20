"""
SemanticsManager for Knowledge Base Loading and Updating.

This module is responsible for locating, loading, and merging semantic
specification files (JSONs) into a unified Knowledge Graph.

Updates:
- Calls `_hydrate_defaults_from_registry` to merge code-defined variants.
- Merges `adapter.definitions` as valid overlays.
- Merges `adapter.import_namespaces` into import maps.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set
from pydantic import ValidationError

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OpDefinition
from ml_switcheroo.frameworks import available_frameworks, get_adapter

# Access Hook Registry to hydrate auto-wired plugins
from ml_switcheroo.core import hooks

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.semantics.merging import (
  merge_tier_data,
  merge_overlay_data,
  infer_tier_from_priority,
)

DISCOVERED_FILENAMES = {"k_discovered.json"}


class SemanticsManager:
  """
  Central database for semantic mappings and configuration.
  """

  def __init__(self):
    self.data: Dict[str, Dict] = {}
    self.import_data: Dict[str, Dict] = {}
    self.framework_configs: Dict[str, Dict] = {}
    self.test_templates: Dict[str, Dict] = {}
    self._known_rng_methods: Set[str] = set()

    self._reverse_index: Dict[str, Tuple[str, Dict]] = {}
    self._key_origins: Dict[str, str] = {}
    self._validation_status: Dict[str, bool] = {}

    # 1. Load Files (Specs & Overlays)
    self._load_knowledge_graph()

    # 2. Hydrate from Registry (Code Overrides)
    self._hydrate_defaults_from_registry()

  def _hydrate_defaults_from_registry(self) -> None:
    """
    Iterates over FrameWork Adapters and Plugin Hooks to extract configuration.
    """
    # --- A. Framework Adapters ---
    for fw_name in available_frameworks():
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      if fw_name not in self.framework_configs:
        self.framework_configs[fw_name] = {}

      # Traits
      if hasattr(adapter, "import_alias") and adapter.import_alias:
        mod_path, alias_name = adapter.import_alias
        self.framework_configs[fw_name]["alias"] = {
          "module": mod_path,
          "name": alias_name,
        }

      if hasattr(adapter, "structural_traits"):
        try:
          traits = adapter.structural_traits
          if traits:
            self.framework_configs[fw_name]["traits"] = traits.model_dump(exclude_unset=True)
        except Exception as e:
          print(f"⚠️ Failed to load structural traits for {fw_name}: {e}")

      if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:
        self.framework_configs[fw_name]["tiers"] = [t.value for t in adapter.supported_tiers]

      if hasattr(adapter, "rng_seed_methods") and adapter.rng_seed_methods:
        self._known_rng_methods.update(adapter.rng_seed_methods)

      # **New: Merge Code-Defined Variants**
      if hasattr(adapter, "definitions"):
        defs = adapter.definitions
        if defs:
          mappings = {k: v.model_dump(exclude_unset=True) for k, v in defs.items()}

          # Pre-label Tiers based on Naming Convention before merging
          # Capitalized ops (Linear, Conv2d) -> Neural
          # Lowercase ops (abs, sum) -> Array
          for op_key in mappings.keys():
            if op_key not in self._key_origins:
              # Simple heuristic: Uppercase usually implies Class/Module (Neural)
              # Lowercase implies Function (Array/Math)
              if op_key[0].isupper():
                self._key_origins[op_key] = SemanticTier.NEURAL.value
              else:
                self._key_origins[op_key] = SemanticTier.ARRAY_API.value

          # Create a virtual snapshot structure
          virtual_snap = {"__framework__": fw_name, "mappings": mappings}
          merge_overlay_data(
            data=self.data,
            key_origins=self._key_origins,
            import_data=self.import_data,
            framework_configs=self.framework_configs,
            test_templates=self.test_templates,
            content=virtual_snap,
            filename=f"{fw_name}_code_defs",
          )

      # **New: Merge Import Namespaces**
      if hasattr(adapter, "import_namespaces"):
        ns_map = adapter.import_namespaces
        if ns_map:
          # Structure imports appropriately: src -> variants -> target -> details
          for src_pkg, details in ns_map.items():
            if src_pkg not in self.import_data:
              self.import_data[src_pkg] = {"variants": {}}
            self.import_data[src_pkg]["variants"][fw_name] = details

      # Wiring Logic (Dynamic/Legacy adjustments)
      if hasattr(adapter, "apply_wiring"):
        dummy_snap = {"__framework__": fw_name}
        try:
          adapter.apply_wiring(dummy_snap)
          merge_overlay_data(
            data=self.data,
            key_origins=self._key_origins,
            import_data=self.import_data,
            framework_configs=self.framework_configs,
            test_templates=self.test_templates,
            content=dummy_snap,
            filename=f"{fw_name}_dynamic_wiring",
          )
        except Exception as e:
          print(f"⚠️ Failed to apply wiring for {fw_name}: {e}")

    # --- B. Auto-Wired Plugins ---
    plugin_metadata = hooks.get_all_hook_metadata()
    for trigger_name, spec in plugin_metadata.items():
      for op_name, op_details in spec.ops.items():
        merge_tier_data(
          data=self.data,
          key_origins=self._key_origins,
          import_data=self.import_data,
          framework_configs=self.framework_configs,
          new_content={op_name: op_details},
          tier=SemanticTier.EXTRAS,
        )

    # Rebuild index after all injections
    self._build_index()

  def get_all_rng_methods(self) -> Set[str]:
    return self._known_rng_methods

  def resolve_variant(self, abstract_id: str, target_fw: str) -> Optional[Dict[str, Any]]:
    defn = self.data.get(abstract_id)
    if not defn:
      return None
    variants = defn.get("variants", {})
    if target_fw in variants:
      return variants[target_fw]

    curr = target_fw
    limit = 5
    while limit > 0:
      parent = None
      config = self.framework_configs.get(curr, {})
      if "extends" in config:
        parent = config["extends"]
      if not parent:
        # Check loaded adapter class inheritance
        # Note: We don't hold references to adapter instances directly map,
        # so we instantiate temporarily or check known config?
        # Since _hydrate_defaults_from_registry ran, it implies adapters exist.
        # We can look up the adapter again here.
        adapter = get_adapter(curr)
        if adapter and hasattr(adapter, "inherits_from") and adapter.inherits_from:
          parent = adapter.inherits_from

      if not parent:
        return None

      if parent in variants:
        return variants[parent]

      curr = parent
      limit -= 1
    return None

  def load_validation_report(self, report_path: Path) -> None:
    if not report_path.exists():
      print(f"⚠️ Validation report not found at {report_path}. Skipping gating.")
      return
    try:
      with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        if isinstance(report, dict):
          self._validation_status.update(report)
          print(f"🔒 Loaded {len(report)} verification statuses.")
    except Exception as e:
      print(f"❌ Error loading validation report: {e}")

  def is_verified(self, abstract_id: str) -> bool:
    status_map = getattr(self, "_validation_status", {})
    return status_map.get(abstract_id, True)

  def get_definition_by_id(self, abstract_id: str) -> Optional[Dict[str, Any]]:
    return self.data.get(abstract_id)

  def get_definition(self, api_name: str) -> Optional[Tuple[str, Dict]]:
    return self._reverse_index.get(api_name)

  def get_known_apis(self) -> Dict[str, Dict]:
    return self.data

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    result = {}
    for src_mod, details in self.import_data.items():
      variants = details.get("variants", {})
      tgt_impl = variants.get(target_fw)

      if tgt_impl:
        target_root = tgt_impl.get("root")
        target_sub = tgt_impl.get("sub")
        alias = tgt_impl.get("alias")
        if target_root:
          result[src_mod] = (target_root, target_sub, alias)
    return result

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    return self.framework_configs.get(framework, {})

  def get_test_template(self, framework: str) -> Optional[Dict[str, str]]:
    return self.test_templates.get(framework)

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    result: Dict[str, Tuple[str, str]] = {}
    for fw, config in self.framework_configs.items():
      alias_conf = config.get("alias")
      if alias_conf and isinstance(alias_conf, dict):
        mod = alias_conf.get("module")
        alias = alias_conf.get("name")
        if mod and alias:
          result[fw] = (mod, alias)
    return result

  def update_definition(self, abstract_id: str, new_data: Dict[str, Any]) -> None:
    try:
      validated = OpDefinition.model_validate(new_data)
      final_data = validated.model_dump(by_alias=True, exclude_unset=True)
    except ValidationError as e:
      print(f"❌ Cannot update invalid definition for '{abstract_id}': {e}")
      return

    self.data[abstract_id] = final_data
    variants = final_data.get("variants", {})
    for _, impl in variants.items():
      if isinstance(impl, dict) and "api" in impl:
        self._reverse_index[impl["api"]] = (abstract_id, final_data)

    tier_str = self._key_origins.get(abstract_id, SemanticTier.ARRAY_API.value)
    filename = "k_array_api.json"
    if tier_str == SemanticTier.NEURAL.value:
      filename = "k_neural_net.json"
    elif tier_str == SemanticTier.EXTRAS.value:
      filename = "k_framework_extras.json"

    file_path = resolve_semantics_dir() / filename
    if file_path.exists():
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          file_content = json.load(f)
      except Exception:
        file_content = {}
    else:
      file_content = {}

    file_content[abstract_id] = final_data
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        json.dump(file_content, f, indent=2, sort_keys=True)
    except Exception as e:
      print(f"❌ Failed to write update for {abstract_id} to {filename}: {e}")

  def _load_knowledge_graph(self) -> None:
    base_path = resolve_semantics_dir()
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

      prioritized_files.sort(key=lambda x: (x[0], x[1].name))

      for priority, fpath in prioritized_files:
        try:
          with open(fpath, "r", encoding="utf-8") as f:
            content = json.load(f)
          tier = infer_tier_from_priority(priority)
          self._merge_tier(content, tier)
        except Exception as e:
          print(f"⚠️ Error loading {fpath.name}: {e}")

    self._load_overlays()
    self._build_index()

  def _load_overlays(self) -> None:
    snap_dir = resolve_snapshots_dir()
    if not snap_dir.exists():
      return
    mapping_files = list(snap_dir.glob("*_map.json"))
    for fpath in mapping_files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)
        self._merge_overlay(content, fpath.name)
      except Exception as e:
        print(f"⚠️ Error loading overlay {fpath.name}: {e}")

  def _merge_overlay(self, content: Dict[str, Any], filename: str) -> None:
    merge_overlay_data(
      data=self.data,
      key_origins=self._key_origins,
      import_data=self.import_data,
      framework_configs=self.framework_configs,
      test_templates=self.test_templates,
      content=content,
      filename=filename,
    )

  def _merge_tier(self, new_data: Dict[str, Any], tier: SemanticTier) -> None:
    merge_tier_data(
      data=self.data,
      key_origins=self._key_origins,
      import_data=self.import_data,
      framework_configs=self.framework_configs,
      new_content=new_data,
      tier=tier,
    )

  def _build_index(self) -> None:
    self._reverse_index.clear()
    for abstract_id, details in self.data.items():
      variants = details.get("variants", {})
      for _engine, impl in variants.items():
        if not impl:
          continue
        api_name = impl.get("api")
        if api_name:
          self._reverse_index[api_name] = (abstract_id, details)
