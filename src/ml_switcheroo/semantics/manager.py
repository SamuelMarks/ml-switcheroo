"""
SemanticsManager for Knowledge Base Loading and Updating.

This module is responsible for locating, loading, and merging semantic
specification files (JSONs) into a unified Knowledge Graph.

It implements a "Hub-and-Spoke" loading strategy:
1.  **Hub (Specs)**: Loads Abstract Operation definitions from `semantics/*.json`
    (e.g., `k_array_api.json`). These define the "Standard" (args, description).
2.  **Spokes (Overlays)**: Scans `snapshots/*_mappings.json` to inject
    framework-specific implementation details (variants) into the abstract definitions.

This separation allows the Specs to remain stable while Framework implementations
evolve rapidly in the snapshots directory.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set, Union
from pydantic import ValidationError

if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  files = None

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OpDefinition
from ml_switcheroo.frameworks import available_frameworks, get_adapter


def resolve_semantics_dir() -> Path:
  """
  Locates the directory containing semantic JSON definitions.

  Prioritizes the local file system (relative to this file) to ensure
  tests and editable installs find the source of truth correctly.
  Falls back to package resources for installed distributions.
  """
  # 1. Local Source Priority (Dev/Test/Editable)
  local_path = Path(__file__).parent
  # Simple check: does the main neural file exist here?
  if (local_path / "k_neural_net.json").exists():
    return local_path

  # 2. Installed Package Fallback
  if sys.version_info >= (3, 9) and files:
    try:
      # Note: wrapping in Path(str(...)) can be brittle with zipped eggs,
      # but standard pip installs extract data or return a path-like object.
      return Path(str(files("ml_switcheroo.semantics")))
    except Exception:
      pass

  return local_path


def resolve_snapshots_dir() -> Path:
  """
  Locates the directory containing framework snapshots and mapping overlays.
  Defaults to the sibling 'snapshots' directory relative to 'semantics'.
  """
  return resolve_semantics_dir().parent / "snapshots"


class SemanticsManager:
  """
  Central database for semantic mappings and configuration.

  Data Source Priority:
  1. Registry Defaults (Code-defined in FrameworkAdapters).
  2. Spec Files (`semantics/*.json`): Defines the Abstract Standards.
  3. Overlay Files (`snapshots/*_mappings.json`): Injects Framework Variants.
  """

  def __init__(self):
    self.data: Dict[str, Dict] = {}
    self.import_data: Dict[str, Dict] = {}

    # Stores framework traits: { "jax": { "alias": {...}, "traits": {...} } }
    self.framework_configs: Dict[str, Dict] = {}

    self.test_templates: Dict[str, Dict] = {}
    self._known_rng_methods: Set[str] = set()

    self._reverse_index: Dict[str, Tuple[str, Dict]] = {}
    self._key_origins: Dict[str, str] = {}
    self._validation_status: Dict[str, bool] = {}

    self._hydrate_defaults_from_registry()
    self._load_knowledge_graph()

  def _hydrate_defaults_from_registry(self) -> None:
    """
    Iterates over all registered frameworks and extracts configuration
    defined in the Adapter code.
    Aggregates:
    - Aliases
    - Test Templates
    - Structural Traits
    - RNG Seed Methods (Purity Analysis)
    """
    for fw_name in available_frameworks():
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      if fw_name not in self.framework_configs:
        self.framework_configs[fw_name] = {}

      # A. Populate Default Aliases
      if hasattr(adapter, "import_alias") and adapter.import_alias:
        mod_path, alias_name = adapter.import_alias
        self.framework_configs[fw_name]["alias"] = {"module": mod_path, "name": alias_name}

      # B. Populate Structural Traits (The Zero-Edit Addition)
      if hasattr(adapter, "structural_traits"):
        try:
          traits = adapter.structural_traits
          # Only serialize if a model is returned and has content
          if traits:
            self.framework_configs[fw_name]["traits"] = traits.model_dump(exclude_unset=True)
        except Exception as e:
          # Log but do not crash initialization if an adapter is malformed
          print(f"⚠️ Failed to load structural traits for {fw_name}: {e}")
      # D. Populate RNG Methods (Purity Analysis)
      if hasattr(adapter, "rng_seed_methods"):
        methods = adapter.rng_seed_methods
        if methods:
          self._known_rng_methods.update(methods)

  def get_all_rng_methods(self) -> Set[str]:
    """Returns the consolidated set of RNG methods from all frameworks."""
    return self._known_rng_methods

  def resolve_variant(self, abstract_id: str, target_fw: str) -> Optional[Dict[str, Any]]:
    """
    Resolves the variant definition for a given target framework, traversing inheritance.
    Priority:
    1. JSON-Defined Inheritance (`__frameworks__` config).
    2. Code-Defined Inheritance (`FrameworkAdapter.inherits_from`).
    """
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

      # 1. JSON Configuration Precedence (Graph defined in semantics takes priority)
      config = self.framework_configs.get(curr, {})
      if "extends" in config:
        parent = config["extends"]

      # 2. Adapter Fallback
      if not parent:
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
        else:
          print(f"❌ Invalid report format in {report_path}. Expected JSON dict.")
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

    # We assume updates go to array api or neural net if known, else extras
    tier_str = self._key_origins.get(abstract_id, SemanticTier.ARRAY_API.value)
    filename = "k_array_api.json"

    if tier_str == SemanticTier.NEURAL.value:
      filename = "k_neural_net.json"
    elif tier_str == SemanticTier.EXTRAS.value:
      filename = "k_framework_extras.json"

    file_path = resolve_semantics_dir() / filename

    # Load existing manually to preserve structure before overwriting
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
    """
    Loads, sorts, and merges all semantic definition files.
    Phase 1: Loads Specs from `semantics/`.
    Phase 2: Loads Framework Overlays from `snapshots/`.
    """
    base_path = resolve_semantics_dir()

    # --- Phase 1: Load Specs ---
    if base_path.exists():
      # 1. Discover all JSONs
      all_files = list(base_path.rglob("*.json"))

      # 2. Assign Priorities based on Tier Heuristics
      prioritized_files: List[Tuple[int, Path]] = []
      for fpath in all_files:
        fname = fpath.name
        priority = 30  # Default (Extras / Extensions)
        if "array" in fname:
          priority = 10
        elif "neural" in fname:
          priority = 20
        elif "templates" in fname:
          priority = 99  # Templates handled separately
        prioritized_files.append((priority, fpath))

      prioritized_files.sort(key=lambda x: (x[0], x[1].name))

      for priority, fpath in prioritized_files:
        try:
          with open(fpath, "r", encoding="utf-8") as f:
            content = json.load(f)

          tier = self._infer_tier(priority)
          self._merge_tier(content, tier)
        except json.JSONDecodeError as e:
          print(f"❌ Error decoding {fpath.name}: {e}")
        except Exception as e:
          print(f"⚠️ Error loading {fpath.name}: {e}")

    # --- Phase 2: Load Overlays ---
    self._load_overlays()

    # --- Phase 3: Final Indexing ---
    self._build_index()

  def _load_overlays(self) -> None:
    """
    Scans the snapshots directory for `*_mappings.json` files and injects
    implementation details into the Loaded Specs.
    """
    snap_dir = resolve_snapshots_dir()
    if not snap_dir.exists():
      return

    # Pattern convention: only load explicit mapping files to key framework-specific moves
    # E.g. torch_mappings.json, jax_mappings.json
    mapping_files = list(snap_dir.glob("*_map.json"))

    for fpath in mapping_files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)
        self._merge_overlay(content, fpath.name)
      except Exception as e:
        print(f"⚠️ Error loading overlay {fpath.name}: {e}")

  def _merge_overlay(self, content: Dict[str, Any], filename: str) -> None:
    """
    Merges a mapping overlay file into the main data.

    Expected JSON Structure:
    {
        "__framework__": "torch",
        "mappings": {
           "Abs": { "api": "torch.abs" },
           "Conv2d": { "api": "torch.nn.Conv2d", "args": {"filters": "out_channels"} }
        },
        "templates": { ... }
    }
    """
    target_fw = content.get("__framework__")

    if not target_fw:
      # Fallback: try to guess from filename 'torch_mappings.json' -> 'torch'
      parts = filename.split("_v")
      if len(parts) > 1:
        target_fw = parts[0]
      else:
        return  # Cannot determine target framework

    # 1. Merge Template Config if present
    if "templates" in content:
      self.test_templates[target_fw] = content["templates"]

    # 2. Merge Framework Traits (Aliases)
    if "framework" in content:
      if target_fw not in self.framework_configs:
        self.framework_configs[target_fw] = content["framework"]
      else:
        self.framework_configs[target_fw].update(content["framework"])

    # 3. Merge Import Maps
    if "imports" in content:
      # Handle inversion: Snapshot has { "torch.nn": { "root": "flax", ... } }
      # Manager structure is { "torch.nn": { "variants": { "jax": { ... } } } }
      for src_mod, details in content["imports"].items():
        if src_mod not in self.import_data:
          self.import_data[src_mod] = {"variants": {}}

        self.import_data[src_mod]["variants"][target_fw] = details

    # 4. Merge Mappings
    mappings = content.get("mappings", {})
    for op_name, implementation in mappings.items():
      # A. Check if Op exists in Spec
      if op_name not in self.data:
        # If not in spec, we can treat it as an extra or create a skeleton.
        # For now, we create a skeleton Extra.
        self.data[op_name] = {
          "description": f"Auto-generated from {filename}",
          "std_args": [],  # Unknown if not in Spec
          "variants": {},
        }
        self._key_origins[op_name] = SemanticTier.EXTRAS.value

      # B. Ensure 'variants' dict exists (Fix for KeyError bug)
      if "variants" not in self.data[op_name]:
        self.data[op_name]["variants"] = {}

      # C. Inject Variant
      # If implementation is explicitly null, it means 'Not Supported'
      # merge logic handles replacing existing variant data
      if implementation is None:
        self.data[op_name]["variants"][target_fw] = None
      else:
        # Merge dictionary to allow augmenting existing data
        if target_fw not in self.data[op_name]["variants"]:
          self.data[op_name]["variants"][target_fw] = {}

        current_variant = self.data[op_name]["variants"][target_fw]
        # Handle case where current_variant might be None (from previous explicit disable)
        if current_variant is None:
          current_variant = {}
          self.data[op_name]["variants"][target_fw] = current_variant

        current_variant.update(implementation)

  def _infer_tier(self, priority: int) -> SemanticTier:
    if priority == 10:
      return SemanticTier.ARRAY_API
    if priority == 20:
      return SemanticTier.NEURAL
    return SemanticTier.EXTRAS

  def _merge_tier(self, new_data: Dict[str, Any], tier: SemanticTier) -> None:
    data_copy = new_data.copy()

    if "__imports__" in data_copy:
      self._merge_imports(data_copy.pop("__imports__"))

    if "__frameworks__" in data_copy:
      self._merge_frameworks(data_copy.pop("__frameworks__"))

    for op_name, details in data_copy.items():
      if op_name in self.data:
        if tier != SemanticTier.EXTRAS:
          prev_tier = self._key_origins.get(op_name, "unknown")
          warnings.warn(
            f"Conflict detected for '{op_name}': Defined in '{prev_tier}' but overwritten by '{tier}' in load.",
            UserWarning,
          )

      try:
        validated_op = OpDefinition.model_validate(details)
        stored_dict = validated_op.model_dump(by_alias=True, exclude_unset=True)
        self.data[op_name] = stored_dict
        self._key_origins[op_name] = tier.value
      except ValidationError as e:
        print(f"⚠️  Skipping invalid definition '{op_name}' in {tier.value}: {e}")
        continue

  def _merge_imports(self, new_imports: Dict[str, Any]) -> None:
    for src_mod, details in new_imports.items():
      if src_mod not in self.import_data:
        self.import_data[src_mod] = details
      else:
        existing_variants = self.import_data[src_mod].get("variants", {})
        new_variants = details.get("variants", {})
        existing_variants.update(new_variants)
        self.import_data[src_mod]["variants"] = existing_variants

  def _merge_frameworks(self, new_configs: Dict[str, Any]) -> None:
    for fw_name, traits in new_configs.items():
      if fw_name not in self.framework_configs:
        self.framework_configs[fw_name] = traits
      else:
        current = self.framework_configs[fw_name]
        if "alias" in traits and "alias" in current:
          current["alias"].update(traits["alias"])
          traits_copy = traits.copy()
          del traits_copy["alias"]
          current.update(traits_copy)
        elif "traits" in traits:
          # If JSON redefined traits, simple merge/overwrite
          current["traits"] = traits["traits"]
        else:
          current.update(traits)

  def _build_index(self) -> None:
    self._reverse_index.clear()
    for abstract_id, details in self.data.items():
      variants = details.get("variants", {})
      for _engine, impl in variants.items():
        # Check if impl is valid (not None)
        if not impl:
          continue
        api_name = impl.get("api")
        if api_name:
          self._reverse_index[api_name] = (abstract_id, details)
