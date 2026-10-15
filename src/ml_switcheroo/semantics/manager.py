"""SemanticsManager for Knowledge Base Loading and Updating.

This module acts as the central database and coordinator for the Semantic
Knowledge Base. It delegates file loading and registry introspection to helper
modules, serving as the primary API for querying operation definitions.

Core Responsibilities:
1.  **State Management**: Holds the merged view of operations, traits, and aliases.
2.  **Querying**: Resolves Abstract Operations to Framework-Specific Variants.
3.  **Coordination**: Triggers file loaders and code hydrators on initialization.
"""

import copy
import json
import logging
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.dsl import PatternDef

# Use base directly to avoid cycle
from ml_switcheroo.frameworks.base import get_adapter

# New modular loaders
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.semantics.registry_loader import RegistryLoader

logger = logging.getLogger(__name__)
_MANAGER_CACHE = None


class SemanticsManager:
  """Central database for semantic mappings and configuration.

  It aggregates data from sources:
  1.  **File System**: JSON Specs (`semantics/`) and Overlays (`snapshots/`).
  2.  **Code Registry**: Python classes (`FrameworkAdapter`, `Plugin`).
  """

  def __init__(self) -> None:
    """Initialize the manager and loads all knowledge sources."""
    global _MANAGER_CACHE
    if _MANAGER_CACHE is not None:
      self.__dict__ = {k: copy.copy(v) for k, v in _MANAGER_CACHE.items()}
      return

    # Core Data Stores
    self.data: dict[Any, Any] = {}
    self.framework_configs: dict[Any, Any] = {}
    self.test_templates: dict[Any, Any] = {}
    self._known_rng_methods: Set[str] = set()
    self.known_magic_args: Set[str] = set()
    self.patterns: List[PatternDef] = []

    # Indexes
    self._reverse_index: dict[Any, Any] = {}
    self._variant_cache: Dict[Tuple[str, str], typing.Optional[dict[str, typing.Any]]] = {}
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

    _MANAGER_CACHE = {k: copy.copy(v) for k, v in self.__dict__.items()}

  def _build_index(self) -> None:
    """Construct the reverse index mapping from concrete API endpoints back to their abstract definitions."""
    from ml_switcheroo.semantics.index_builder import IndexBuilder

    IndexBuilder.build_index(self)

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Generate the import mapping for the ImportFixer based on Tier linking.

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
    """Find parent framework key if exists.

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

  def get_framework_ecosystem(self, fw: str) -> typing.Set[str]:
    """Recursively collect the framework and all its ancestors' root module aliases.

    Args:
        fw: The framework name (e.g. 'flax_nnx').

    Returns:
        A set of root module names (e.g. {'flax_nnx', 'flax', 'jax'}).
    """
    ecosystem = {fw}
    current: typing.Optional[str] = fw

    while current:
      # Add the alias module root if defined
      conf = self.get_framework_config(current)
      alias_module = conf.get("alias", {}).get("module", "")
      if alias_module:
        ecosystem.add(alias_module.split(".")[0])

      # Traverse up the inheritance tree
      current = self._resolve_inheritance(current)
      if current:
        ecosystem.add(current)

    return ecosystem - {""}

  def resolve_variant(self, abstract_id: str, target_fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolve the implementation of an abstract operation.

    Args:
        abstract_id: Canonical abstract operation identifier.
        target_fw: Target framework key.

    Returns:
        Target variant dictionary if found, else None.
    """
    from ml_switcheroo.semantics.resolution import ResolutionEngine

    return ResolutionEngine.resolve_variant(self, abstract_id, target_fw)

  def is_verified(self, abstract_id: str) -> bool:
    """Return True if the operation is marked verified (or untracked).

    Args:
        abstract_id: The unique identifier of the abstract operation.

    Returns:
        True if the operation is verified or untracked, False otherwise.
    """
    status_map = getattr(self, "_validation_status", {})
    return bool(status_map.get(abstract_id, True))

  def get_definition_by_id(self, abstract_id: str) -> typing.Optional[dict[str, typing.Any]]:
    """Direct dictionary access.

    Args:
        abstract_id: The unique identifier of the abstract operation.

    Returns:
        The dictionary containing the definition of the abstract operation, or
        None if not found.
    """
    dt: dict[str, typing.Any] = self.data
    return dt.get(abstract_id)

  def get_definition(self, api_name: str) -> typing.Optional[typing.Tuple[str, dict[str, typing.Any]]]:
    """Reverse lookup from concrete API string or Abstract ID fallback.

    Args:
        api_name: Concrete API path or abstract operation ID.

    Returns:
        Tuple of `(abstract_id, details)` if found, else None.
    """
    from ml_switcheroo.semantics.resolution import ResolutionEngine

    return ResolutionEngine.get_definition(self, api_name)

  def get_known_apis(self) -> typing.Dict[str, dict[str, typing.Any]]:
    """Return full knowledge graph.

    Returns:
        A dictionary mapping abstract IDs to their full operation definitions.
    """
    dt: dict[str, typing.Any] = self.data
    return dt

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Return definition of framework traits.

    Args:
        framework: The name of the target framework.

    Returns:
        A dictionary representing the framework configuration and traits.
    """
    d_traits: dict[str, typing.Any] = self.framework_configs.get(framework, {})
    return d_traits

  def get_test_template(self, framework: str) -> Optional[Dict[str, str]]:
    """Return testing codegen templates.

    Args:
        framework: The name of the target framework.

    Returns:
        A dictionary mapping template names to template strings if available,
        or None.
    """
    return self.test_templates.get(framework)

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    """Return a map of {fw: (module, alias)}.

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
    """Return aggregate list of random seeding methods.

    Returns:
        A set of random number generator method names.
    """
    return self._known_rng_methods

  def get_patterns(self) -> List[PatternDef]:
    """Return the list of loaded fusion patterns.

    Returns:
        A list of PatternDef objects defining fusion patterns.
    """
    return self.patterns

  def load_validation_report(self, report_path: Path) -> None:
    """Load a CI verification report to gate unavailable operations.

    Args:
        report_path: The filesystem path to the validation report JSON file.
    """
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

  def update_definition(self, abstract_id: str, new_data: dict[str, typing.Any]) -> None:
    """Update an operation definition in memory and persists to disk.

    Args:
        abstract_id: Abstract operation identifier to update.
        new_data: New operation definition dictionary to validate and persist.
    """
    from ml_switcheroo.semantics.validator import SemanticsValidator

    final_data = SemanticsValidator.validate_and_dump(abstract_id, new_data)
    if not final_data:
      return

    self.data[abstract_id] = final_data
    variants = final_data.get("variants", {})
    for _, impl in variants.items():
      if isinstance(impl, dict) and "api" in impl:
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

  def lookup_snapshot_symbol(self, framework: str, symbol: str) -> typing.Optional[dict[str, typing.Any]]:
    """Look up a grounded symbol from indexed framework snapshots.

    Args:
        framework: The framework target identifier (e.g. 'torch', 'jax').
        symbol: The fully qualified or base symbol name.

    Returns:
        The snapshot dictionary for the symbol if found, or None.
    """
    try:
      from ml_ecosystem_snapshots.index import lookup_symbol

      res = lookup_symbol(framework, symbol)
      if isinstance(res, dict):
        return res
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Snapshot symbol lookup failed for %s.%s: %s", framework, symbol, exc)
    return None

  def search_snapshot_symbols(
    self,
    query: str,
    framework: typing.Optional[str] = None,
    limit: int = 10,
  ) -> typing.List[dict[str, typing.Any]]:
    """Search for symbols in indexed snapshots using full-text search.

    Args:
        query: Search query or symbol prefix.
        framework: Optional framework target to restrict the search.
        limit: Maximum number of search results to return.

    Returns:
        A list of matching symbol dictionaries from the snapshot index.
    """
    try:
      from ml_ecosystem_snapshots.index import search_index

      res = search_index(query, framework=framework, limit=limit)
      if isinstance(res, list):
        return res
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Snapshot symbol search failed for query '%s': %s", query, exc)
    return []

  def validate_python_call(
    self,
    framework: str,
    api_path: str,
    args: typing.Optional[typing.List[typing.Any]] = None,
    kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
  ) -> typing.Any:
    """Validate a Python call against grounded framework snapshot specifications.

    Args:
        framework: Target framework identifier (e.g. 'torch', 'jax').
        api_path: Full or base API path (e.g. 'torch.add').
        args: Optional list of positional argument representations.
        kwargs: Optional dictionary of keyword argument representations.

    Returns:
        A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.python_fw import validate_python_call

      return validate_python_call(
        framework=framework,
        api_path=api_path,
        args=args or [],
        kwargs=kwargs or {},
      )
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Snapshot Python call validation failed for %s.%s: %s", framework, api_path, exc)
      return None

  def diff_snapshot_versions(self, snap1: typing.Any, snap2: typing.Any) -> typing.Any:
    """Diff two snapshots to track API modifications, deprecations, and additions.

    Args:
        snap1: The older snapshot data structure or path.
        snap2: The newer snapshot data structure or path.

    Returns:
        A DiffResult object from ml_ecosystem_snapshots if available, or None.
    """
    try:
      from ml_ecosystem_snapshots.diff import diff_snapshots

      return diff_snapshots(snap1, snap2)
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Snapshot diff failed: %s", exc)
      return None

  def generate_snapshot_changelog(self, diff: typing.Any) -> str:
    """Generate a Markdown changelog describing differences between snapshot versions.

    Args:
        diff: A DiffResult object or structured difference report.

    Returns:
        A formatted markdown string detailing API changes, or empty string on failure.
    """
    try:
      from ml_ecosystem_snapshots.diff import generate_changelog

      return str(generate_changelog(diff))
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Snapshot changelog generation failed: %s", exc)
      return ""

  def extract_framework_isolated(
    self,
    framework: str,
    include_nonpublic: bool = False,
    timeout: int = 300,
  ) -> typing.Dict[str, typing.Any]:
    """Extract a framework snapshot dynamically within an isolated child subprocess.

    Prevents CUDA/Metal memory pre-allocation locks and C-extension segfaults
    from terminating the parent process or test suite.

    Args:
        framework: Name of the framework target (e.g. 'torch', 'jax', 'mlx').
        include_nonpublic: Whether to include non-public APIs in extraction.
        timeout: Maximum duration in seconds to wait for child process.

    Returns:
        Dictionary containing extracted snapshot metadata and items, or empty dict on failure.
    """
    try:
      from ml_ecosystem_snapshots.api import extract_snapshot_isolated

      res = extract_snapshot_isolated(
        framework=framework,
        include_nonpublic=include_nonpublic,
        timeout=timeout,
      )
      if isinstance(res, dict):
        return res
    except (ImportError, OSError, ValueError, RuntimeError, Exception) as exc:
      logger.debug("Isolated snapshot extraction failed for %s: %s", framework, exc)
    return {}
