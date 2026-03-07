"""
SemanticsManager for Knowledge Base Loading and Updating.

This module acts as the central database and coordinator for the Semantic
Knowledge Base. It delegates file loading and registry introspection to helper
modules, serving as the primary API for querying operation definitions.

Core Responsibilities:
1.  **State Management**: Holds the merged view of operations, traits, and aliases.
2.  **Querying**: Resolves Abstract Operations to Framework-Specific Variants.
3.  **Coordination**: Triggers file loaders and code hydrators on initialization.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Set, List
from pydantic import ValidationError

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OperationDef, PatternDef
from ml_switcheroo.semantics.paths import resolve_semantics_dir

# Use base directly to avoid cycle
from ml_switcheroo.frameworks.base import get_adapter

# New modular loaders
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


class SemanticsManager:
  """
  Central database for semantic mappings and configuration.

  It aggregates data from sources:
  1.  **File System**: JSON Specs (`semantics/`) and Overlays (`snapshots/`).
  2.  **Code Registry**: Python classes (`FrameworkAdapter`, `Plugin`).
  """

  def __init__(self) -> None:
    """Initializes the manager and loads all knowledge sources."""
    # Core Data Stores
    self.data: Dict[str, Dict] = {}
    self.framework_configs: Dict[str, Dict] = {}
    self.test_templates: Dict[str, Dict] = {}
    self._known_rng_methods: Set[str] = set()
    self.known_magic_args: Set[str] = set()
    self.patterns: List[PatternDef] = []

    # Indexes
    self._reverse_index: Dict[str, Tuple[str, Dict]] = {}
    self._key_origins: Dict[str, str] = {}
    self._validation_status: Dict[str, bool] = {}

    # Import Abstraction
    # Map[Framework, Map[Tier, NamespaceConfig]]
    self._providers: Dict[str, Dict[SemanticTier, Dict[str, str]]] = {}
    # Map[ImportPath, Tuple[Framework, Tier]]
    self._source_registry: Dict[str, Tuple[str, SemanticTier]] = {}

    # --- Phase 1: File System Loading (Hub & Spokes) ---
    file_loader = KnowledgeBaseLoader(self)
    file_loader.load_knowledge_graph()

    # --- Phase 2: Registry Hydration ---
    registry_loader = RegistryLoader(self)
    registry_loader.hydrate()

    # --- Phase 3: Indexing ---
    self._build_index()

  def _build_index(self) -> None:
    """
    Constructs the reverse index mapping from concrete API endpoints
    back to their abstract definitions.
    """
    self._reverse_index.clear()
    alias_map = {}
    for fw, config in self.framework_configs.items():
      if "alias" in config:
        mod = config["alias"].get("module")
        name = config["alias"].get("name")
        if mod and name:
          alias_map[name] = mod

    alias_map["tf"] = "tensorflow"
    alias_map["jnp"] = "jax.numpy"
    alias_map["np"] = "numpy"
    alias_map["mx"] = "mlx.core"
    alias_map["nn"] = "torch.nn"

    def get_priority(abs_id, details, tier):
      """
      Determines indexing priority when multiple abstract ops map to the same target API.
      This handles overlaps between generic ops like `cat` vs `concat`.
      """
      score = 0
      if abs_id == "cat":
        score += 1000
      elif abs_id == "Append":
        score -= 1000
      elif abs_id == "concat":
        score -= 500

      if abs_id == "Mean":
        score += 1000
      elif abs_id == "Average":
        score -= 1000
      elif abs_id == "mean":
        score -= 500

      if abs_id == "relu":
        score += 100
      elif abs_id == "ReLU":
        score -= 100

      if abs_id == "MultiHeadAttention":
        score += 1000
      elif abs_id == "AttentionLayer":
        score -= 1000

      if abs_id == "Dropout":
        score += 1000
      elif abs_id == "Dropout_":
        score -= 1000

      if tier == SemanticTier.ARRAY_API.value:
        score += 50
      elif tier == SemanticTier.NEURAL.value:
        score -= 50

      score += len(details.get("variants", {}))
      return score

    for abstract_id, details in self.data.items():
      variants = details.get("variants", {})
      tier = self._key_origins.get(abstract_id)
      score = get_priority(abstract_id, details, tier)

      for _engine, impl in variants.items():
        if not impl:
          continue
        api_name = impl.get("api")
        if api_name:

          def register_api(name):
            """
            Registers the target concrete API mapped back to its abstract concept.
            Uses tie-breaker scores when overlaps are found.
            """
            if name in self._reverse_index:
              existing_id, existing_details = self._reverse_index[name]
              existing_tier = self._key_origins.get(existing_id)
              existing_score = get_priority(existing_id, existing_details, existing_tier)
              if score > existing_score:
                self._reverse_index[name] = (abstract_id, details)
            else:
              self._reverse_index[name] = (abstract_id, details)

          register_api(api_name)

          parts = api_name.split(".")
          if parts[0] in alias_map:
            fqn = alias_map[parts[0]] + "." + ".".join(parts[1:])
            register_api(fqn)

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """
    Generates the import mapping for the ImportFixer based on Tier linking.

    Args:
        target_fw: The framework being targeted.

    Returns:
        Dict mapping source import paths to (root, sub, alias) tuples.
    """
    result = {}
    target_providers = self._providers.get(target_fw, {})

    parent = self._resolve_inheritance(target_fw)
    parent_providers = self._providers.get(parent, {}) if parent else {}

    for src_path, (_, tier) in self._source_registry.items():
      target_config = target_providers.get(tier)

      if not target_config:
        target_config = parent_providers.get(tier)

      if target_config:
        root = target_config.get("root")
        sub = target_config.get("sub")
        alias = target_config.get("alias")

        if root:
          result[src_path] = (root, sub, alias)

    return result

  def _resolve_inheritance(self, fw: str) -> Optional[str]:
    """Finds parent framework key if exists."""
    conf = self.framework_configs.get(fw, {})
    if "extends" in conf:
      return conf["extends"]

    adapter = get_adapter(fw)
    if adapter and hasattr(adapter, "inherits_from"):
      return adapter.inherits_from
    return None  # pragma: no cover

  def resolve_variant(self, abstract_id: str, target_fw: str) -> Optional[Dict[str, Any]]:
    """Resolves the implementation of an abstract operation."""
    defn = self.data.get(abstract_id)
    if not defn:
      return None  # pragma: no cover
    variants = defn.get("variants", {})
    if target_fw in variants:
      return variants[target_fw]

    curr = target_fw
    limit = 5
    while limit > 0:
      parent = self._resolve_inheritance(curr)
      if not parent:
        return None
      if parent in variants:
        return variants[parent]
      curr = parent
      limit -= 1
    return None

  def is_verified(self, abstract_id: str) -> bool:
    """Returns True if the operation is marked verified (or untracked)."""
    status_map = getattr(self, "_validation_status", {})
    return status_map.get(abstract_id, True)

  def get_definition_by_id(self, abstract_id: str) -> Optional[Dict[str, Any]]:
    """Direct dictionary access."""
    return self.data.get(abstract_id)

  def get_definition(self, api_name: str) -> Optional[Tuple[str, Dict]]:
    """Reverse lookup from concrete API string or Abstract ID fallback."""
    res = self._reverse_index.get(api_name)
    if res:
      return res

    if api_name in self.data:
      return (api_name, self.data[api_name])

    return None

  def get_known_apis(self) -> Dict[str, Dict]:
    """Returns full knowledge graph."""
    return self.data

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Returns definition of framework traits."""
    return self.framework_configs.get(framework, {})

  def get_test_template(self, framework: str) -> Optional[Dict[str, str]]:
    """Returns testing codegen templates."""
    return self.test_templates.get(framework)

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    """Returns a map of {fw: (module, alias)}."""
    result: Dict[str, Tuple[str, str]] = {}
    for fw, config in self.framework_configs.items():
      alias_conf = config.get("alias")
      if alias_conf and isinstance(alias_conf, dict):
        mod = alias_conf.get("module")
        alias = alias_conf.get("name")
        if mod and alias:
          result[fw] = (mod, alias)
    return result

  def get_all_rng_methods(self) -> Set[str]:
    """Returns aggregate list of random seeding methods."""
    return self._known_rng_methods  # pragma: no cover

  def get_patterns(self) -> List[PatternDef]:
    """Returns the list of loaded fusion patterns."""
    return self.patterns

  def load_validation_report(self, report_path: Path) -> None:
    """Loads a CI verification report to gate unavailable operations."""
    if not report_path.exists():
      print(f"⚠️ Validation report not found at {report_path}. Skipping gating.")  # pragma: no cover
      return  # pragma: no cover
    try:
      with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        if isinstance(report, dict):
          self._validation_status.update(report)
          print(f"🔒 Loaded {len(report)} verification statuses.")
    except Exception as e:  # pragma: no cover
      print(f"❌ Error loading validation report: {e}")  # pragma: no cover

  def update_definition(self, abstract_id: str, new_data: Dict[str, Any]) -> None:
    """Updates an operation definition in memory and persists to disk."""
    # Create a copy to inject defaults without mutating input
    details_to_validate = new_data.copy()

    # 1. Inject missing fields required by Schema if not present
    if "operation" not in details_to_validate:
      details_to_validate["operation"] = abstract_id
    if "variants" not in details_to_validate:
      details_to_validate["variants"] = {}  # pragma: no cover
    if "description" not in details_to_validate:
      details_to_validate["description"] = f"Definition for {abstract_id}"
    if "std_args" not in details_to_validate:
      details_to_validate["std_args"] = []

    try:
      validated = OperationDef.model_validate(details_to_validate)
      final_data = validated.model_dump(by_alias=True, exclude_unset=True)
    except ValidationError as e:  # pragma: no cover
      print(f"❌ Cannot update invalid definition for '{abstract_id}': {e}")  # pragma: no cover
      return  # pragma: no cover

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
      filename = "k_framework_extras.json"  # pragma: no cover

    file_path = resolve_semantics_dir() / filename
    if file_path.exists():
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          file_content = json.load(f)
      except Exception:  # pragma: no cover
        file_content = {}  # pragma: no cover
    else:
      file_content = {}  # pragma: no cover

    file_content[abstract_id] = final_data
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        json.dump(file_content, f, indent=2, sort_keys=True)
    except Exception as e:  # pragma: no cover
      print(f"❌ Failed to write update for {abstract_id} to {filename}: {e}")  # pragma: no cover
