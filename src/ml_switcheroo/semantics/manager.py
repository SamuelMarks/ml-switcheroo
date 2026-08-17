"""SemanticsManager for Knowledge Base Loading and Updating.

This module acts as the central database and coordinator for the Semantic
Knowledge Base. It delegates file loading and registry introspection to helper
modules, serving as the primary API for querying operation definitions.

Core Responsibilities:
1.  **State Management**: Holds the merged view of operations, traits, and aliases.
2.  **Querying**: Resolves Abstract Operations to Framework-Specific Variants.
3.  **Coordination**: Triggers file loaders and code hydrators on initialization.
"""

from typing import Any

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List
from pydantic import ValidationError

from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.core.dsl import OperationDef, PatternDef
from ml_switcheroo.semantics.paths import resolve_semantics_dir

# Use base directly to avoid cycle
from ml_switcheroo.frameworks.base import get_adapter

# New modular loaders
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


class SemanticsManager:
  """Central database for semantic mappings and configuration.

  It aggregates data from sources:
  1.  **File System**: JSON Specs (`semantics/`) and Overlays (`snapshots/`).
  2.  **Code Registry**: Python classes (`FrameworkAdapter`, `Plugin`).
  """

  def __init__(self) -> None:
    """Initializes the manager and loads all knowledge sources."""
    # Core Data Stores
    self.data: Dict[str, Dict[Any, Any]] = {}
    self.framework_configs: Dict[str, Dict[Any, Any]] = {}
    self.test_templates: Dict[str, Dict[Any, Any]] = {}
    self._known_rng_methods: Set[str] = set()
    self.known_magic_args: Set[str] = set()
    self.patterns: List[PatternDef] = []

    # Indexes
    self._reverse_index: Dict[str, Tuple[str, Dict[Any, Any]]] = {}
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
    """Constructs the reverse index mapping from concrete API endpoints back to their abstract definitions."""
    self._reverse_index.clear()
    alias_map = {}

    aliases_json_path = os.path.join(os.path.dirname(__file__), "aliases.json")
    if os.path.exists(aliases_json_path):  # pragma: no branch
      with open(aliases_json_path, "r", encoding="utf-8") as f:
        alias_map.update(json.load(f))

    for fw, config in self.framework_configs.items():
      if "alias" in config:
        mod = config["alias"].get("module")
        name = config["alias"].get("name")
        if mod and name:  # pragma: no branch
          alias_map[name] = mod

    priority_scores = {}
    priority_json_path = os.path.join(os.path.dirname(__file__), "priority_scores.json")
    if os.path.exists(priority_json_path):  # pragma: no branch
      with open(priority_json_path, "r", encoding="utf-8") as f:
        priority_scores = json.load(f)

    def get_priority(abs_id: Any, details: Any, tier: Any) -> Any:
      """Determines indexing priority when multiple abstract ops map to the same target API.

      This handles overlaps between generic ops like `cat` vs `concat`.

      Args:
          abs_id: The abstract operation identifier.
          details: The details of the abstract operation.
          tier: The semantic tier of the abstract operation.

      Returns:
          An integer priority score (higher score wins).
      """
      score = priority_scores.get(abs_id, 0)

      if tier == SemanticTier.ARRAY_API.value:
        score += 50
      elif tier == SemanticTier.NEURAL.value:
        score -= 50
      elif tier == SemanticTier.EXTRAS.value:
        score -= 100

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

          def register_api(name: Any) -> Any:
            """Registers the target concrete API mapped back to its abstract concept.

            Uses tie-breaker scores when overlaps are found.

            Args:
                name: The concrete API name or fully qualified name to register.
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

          # Forward mapping: short to long (e.g. tf.abs -> tensorflow.abs)
          parts = api_name.split(".")
          if parts[0] in alias_map:
            fqn = alias_map[parts[0]] + "." + ".".join(parts[1:])
            register_api(fqn)

          # Reverse mapping: long to short (e.g. torch.nn.Conv2d -> nn.Conv2d)
          for alias_name, module_path in alias_map.items():
            if api_name.startswith(module_path + "."):
              fqn = alias_name + api_name[len(module_path) :]
              register_api(fqn)

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Generates the import mapping for the ImportFixer based on Tier linking.

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

        if root:  # pragma: no branch
          result[src_path] = (root, sub, alias)

    return result

  def _resolve_inheritance(self, fw: str) -> Optional[str]:
    """Finds parent framework key if exists.

    Args:
        fw: The framework name/key to check.

    Returns:
        The parent framework key as a string, or None if no inheritance exists.
    """
    conf = self.framework_configs.get(fw, {})
    if "extends" in conf:
      return str(conf["extends"])

    adapter = get_adapter(fw)
    if adapter and hasattr(adapter, "inherits_from"):
      return adapter.inherits_from
    return None

  def resolve_variant(self, abstract_id: str, target_fw: str) -> Optional[Dict[str, Any]]:
    """Resolves the implementation of an abstract operation.

    Args:
        abstract_id: The unique identifier of the abstract operation.
        target_fw: The target framework name to resolve the variant for.

    Returns:
        A dictionary containing the implementation variant details, or None if
        the variant cannot be resolved.
    """
    defn = self.data.get(abstract_id)
    if not defn:
      return None
    variants = defn.get("variants", {})
    if target_fw in variants:
      return variants[target_fw]  # type: ignore

    curr = target_fw
    limit = 5
    while limit > 0:
      parent = self._resolve_inheritance(curr)
      if not parent:
        return None
      if parent in variants:
        return variants[parent]  # type: ignore
      curr = parent
      limit -= 1
    return None

  def is_verified(self, abstract_id: str) -> bool:
    """Returns True if the operation is marked verified (or untracked).

    Args:
        abstract_id: The unique identifier of the abstract operation.

    Returns:
        True if the operation is verified or untracked, False otherwise.
    """
    status_map = getattr(self, "_validation_status", {})
    return status_map.get(abstract_id, True)  # type: ignore

  def get_definition_by_id(self, abstract_id: str) -> Optional[Dict[str, Any]]:
    """Direct dictionary access.

    Args:
        abstract_id: The unique identifier of the abstract operation.

    Returns:
        The dictionary containing the definition of the abstract operation, or
        None if not found.
    """
    return self.data.get(abstract_id)

  def get_definition(self, api_name: str) -> Optional[Tuple[str, Dict[Any, Any]]]:
    """Reverse lookup from concrete API string or Abstract ID fallback.

    Args:
        api_name: The concrete API endpoint string or abstract ID fallback.

    Returns:
        A tuple of (abstract_id, definition_dict) if resolved, or None if not
        found.
    """
    res = self._reverse_index.get(api_name)
    if res:
      return res

    if api_name in self.data:
      return (api_name, self.data[api_name])

    return None

  def get_known_apis(self) -> Dict[str, Dict[Any, Any]]:
    """Returns full knowledge graph.

    Returns:
        A dictionary mapping abstract IDs to their full operation definitions.
    """
    return self.data

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Returns definition of framework traits.

    Args:
        framework: The name of the target framework.

    Returns:
        A dictionary representing the framework configuration and traits.
    """
    return self.framework_configs.get(framework, {})

  def get_test_template(self, framework: str) -> Optional[Dict[str, str]]:
    """Returns testing codegen templates.

    Args:
        framework: The name of the target framework.

    Returns:
        A dictionary mapping template names to template strings if available,
        or None.
    """
    return self.test_templates.get(framework)

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    """Returns a map of {fw: (module, alias)}.

    Returns:
        A dictionary mapping framework identifiers to a tuple containing the
        importable module path and its designated import alias.
    """
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
    """Returns aggregate list of random seeding methods.

    Returns:
        A set of random number generator method names.
    """
    return self._known_rng_methods

  def get_patterns(self) -> List[PatternDef]:
    """Returns the list of loaded fusion patterns.

    Returns:
        A list of PatternDef objects defining fusion patterns.
    """
    return self.patterns

  def load_validation_report(self, report_path: Path) -> None:
    """Loads a CI verification report to gate unavailable operations.

    Args:
        report_path: The filesystem path to the validation report JSON file.
    """
    if not report_path.exists():
      print(f"⚠️ Validation report not found at {report_path}. Skipping gating.")
      return
    try:
      with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        if isinstance(report, dict):  # pragma: no branch
          self._validation_status.update(report)
          print(f"🔒 Loaded {len(report)} verification statuses.")
    except Exception as e:
      print(f"❌ Error loading validation report: {e}")

  def update_definition(self, abstract_id: str, new_data: Dict[str, Any]) -> None:
    """Updates an operation definition in memory and persists to disk.

    Args:
        abstract_id: The unique identifier of the abstract operation.
        new_data: The dictionary containing the updated operation details.
    """
    # Create a copy to inject defaults without mutating input
    details_to_validate = new_data.copy()

    # 1. Inject missing fields required by Schema if not present
    if "operation" not in details_to_validate:
      details_to_validate["operation"] = abstract_id
    if "variants" not in details_to_validate:
      details_to_validate["variants"] = {}
    if "description" not in details_to_validate:
      details_to_validate["description"] = f"Definition for {abstract_id}"
    if "std_args" not in details_to_validate:
      details_to_validate["std_args"] = []

    try:
      validated = OperationDef.model_validate(details_to_validate)
      final_data = validated.model_dump(by_alias=True, exclude_unset=True)
    except ValidationError as e:
      print(f"❌ Cannot update invalid definition for '{abstract_id}': {e}")
      return

    self.data[abstract_id] = final_data
    variants = final_data.get("variants", {})
    for _, impl in variants.items():
      if isinstance(impl, dict) and "api" in impl:  # pragma: no branch
        self._reverse_index[impl["api"]] = (abstract_id, final_data)

    safe_name = abstract_id.replace("/", "_")
    filename = f"{safe_name}.yaml"
    odl_dir = resolve_semantics_dir() / "odl"
    odl_dir.mkdir(parents=True, exist_ok=True)
    file_path = odl_dir / filename

    try:
      with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(final_data, f, sort_keys=False, indent=2, default_flow_style=False)
    except Exception as e:
      print(f"❌ Failed to write update for {abstract_id} to {filename}: {e}")
