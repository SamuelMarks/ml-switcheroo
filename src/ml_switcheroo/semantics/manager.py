"""
SemanticsManager for Knowledge Base Loading and Updating.

This module is responsible for locating, loading, and merging semantic
specification files (JSONs) into a unified Knowledge Graph.

It supports **Distributed Semantics**, allowing definitions to be split across
multiple files and directories (e.g., `semantics/extensions/*.json`).
Files are loaded recursively and merged with a tier-based priority system.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set
from pydantic import ValidationError

if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  files = None

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OpDefinition
from ml_switcheroo.frameworks import available_frameworks, get_adapter


def resolve_semantics_dir() -> Path:
  if sys.version_info >= (3, 9) and files:
    return Path(str(files("ml_switcheroo.semantics")))
  return Path(__file__).parent


class SemanticsManager:
  """
  Central database for semantic mappings and configuration.

  Data Source Priority:
  1. Registry Defaults (Code-defined in FrameworkAdapters).
  2. JSON Files (User/System defined in `semantics/**/*.json`).
     Files are loaded in the following order of precedence (Lowest to Highest):
     - Array API (Math)
     - Neural Net (Layers)
     - Extensions/Extras (Patches & New Frameworks)
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

      # C. Populate Test Templates
      if (
        hasattr(adapter, "get_import_stmts")
        and hasattr(adapter, "get_creation_syntax")
        and hasattr(adapter, "get_numpy_conversion_syntax")
      ):
        self.test_templates[fw_name] = {
          "import": adapter.get_import_stmts(),
          "convert_input": adapter.get_creation_syntax("{np_var}"),
          "to_numpy": adapter.get_numpy_conversion_syntax("{res_var}"),
        }

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
    Supports distributed semantics via recursive globbing.
    """
    base_path = resolve_semantics_dir()
    if not base_path.exists():
      return

    # 1. Discover all JSONs
    all_files = list(base_path.rglob("*.json"))

    # 2. Assign Priorities based on Tier Heuristics
    # Lower priority loads first, Higher priority overrides.
    # Standard Order: Array (10) -> Neural (20) -> Extras/Extensions (30)
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

    # 3. Sort by Priority then Alphabetically
    prioritized_files.sort(key=lambda x: (x[0], x[1].name))

    # 4. Load Loop
    for priority, fpath in prioritized_files:
      # Separate handling for test templates
      if "test_templates" in fpath.name:
        self._load_templates_file(fpath)
        continue

      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)

        tier = self._infer_tier(priority)
        self._merge_tier(content, tier)
      except json.JSONDecodeError as e:
        print(f"❌ Error decoding {fpath.name}: {e}")
      except Exception as e:
        print(f"⚠️ Error loading {fpath.name}: {e}")

    self._build_index()

  def _load_templates_file(self, fpath: Path) -> None:
    try:
      with open(fpath, "r", encoding="utf-8") as f:
        content = json.load(f)
      self._merge_templates(content)
    except Exception as e:
      print(f"⚠️ Failed to load templates from {fpath.name}: {e}")

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

    if "__templates__" in data_copy:
      self._merge_templates(data_copy.pop("__templates__"))

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

  def _merge_templates(self, new_templates: Dict[str, Any]) -> None:
    for fw_name, traits in new_templates.items():
      if fw_name not in self.test_templates:
        self.test_templates[fw_name] = traits
      else:
        self.test_templates[fw_name].update(traits)

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
