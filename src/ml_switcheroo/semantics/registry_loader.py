"""Registry Hydration Logic.

This module introspects the active Python environment to extract semantic definitions
from registered Framework Adapters and Plugin Hooks. This enables "Code-First"
definitions alongside "Config-First" JSONs.
"""

import logging
import typing

from pydantic import JsonValue

import ml_switcheroo

if typing.TYPE_CHECKING:
  import ml_switcheroo.semantics.manager

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core import hooks_registry as _hooks_registry
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.frameworks.base import FrameworkAdapter, ImportConfig, available_frameworks, get_adapter
from ml_switcheroo.semantics.merging import (
  JsonValueLike,
  TypedSemanticsStore,
  merge_overlay_data,
  merge_tier_data,
  merge_tier_with_models,
)
from ml_switcheroo.semantics.schema import FrameworkAlias, FrameworkTraits, PluginTraits, StructuralTraits

hooks = _hooks_registry

logger = logging.getLogger(__name__)


class RegistryLoader:
  """Hydrate the SemanticsManager from python objects (Adapters/Plugins)."""

  def __init__(self, manager: "ml_switcheroo.semantics.manager.SemanticsManager") -> None:
    """Initialize the loader.

    Args:
        manager: The parent SemanticsManager instance.
    """
    self.mgr = manager

  def hydrate(self) -> None:
    """Execute entry point. Scans adapters and plugins.

    Returns:
        None
    """
    self._hydrate_adapters()
    self._hydrate_plugins()
    self.index_variants()

  def _typed_store(self) -> TypedSemanticsStore | None:
    """Return a ``TypedSemanticsStore`` bound to ``self.mgr`` when typed dictionaries exist.

    Returns:
        ``TypedSemanticsStore`` instance if ``self.mgr`` exposes typed stores, else ``None``.
    """
    try:
      ops = self.mgr.operations
      fw_traits = self.mgr.framework_traits
    except AttributeError:
      return None
    if type(ops) is not dict or type(fw_traits) is not dict:
      return None
    return TypedSemanticsStore(operations=ops, framework_traits=fw_traits)

  def index_variants(self) -> None:
    """Index operation variants into hash maps for O(1) lookups.

    Returns:
        None
    """
    try:
      self.mgr._variant_cache.clear()
      for abstract_id, details in self.mgr.data.items():
        if isinstance(details, dict):
          variants = details.get("variants", {})
          if isinstance(variants, dict):
            for fw_name, variant in variants.items():
              if isinstance(variant, dict):
                self.mgr._variant_cache[(abstract_id, fw_name)] = variant
    except AttributeError:
      pass

    try:
      typed_cache = self.mgr._typed_variant_cache
      ops = self.mgr.operations
      if type(typed_cache) is dict and type(ops) is dict:
        typed_cache.clear()
        for abstract_id, op_model in ops.items():
          for fw_name, variant_model in op_model.variants.items():
            if variant_model is not None:
              typed_cache[(abstract_id, fw_name)] = variant_model
    except AttributeError:
      pass

  def _hydrate_adapters(self) -> None:
    """Iterate over registered FrameworkAdapters to extract Traits and Mappings.

    Returns:
        None
    """
    for fw_name in available_frameworks():
      adapter = get_adapter(fw_name)
      if not adapter:
        continue

      # Ensure config dict exists
      if fw_name not in self.mgr.framework_configs:
        self.mgr.framework_configs[fw_name] = {}

      # 1. Load Structural Traits & Metadata
      self._load_adapter_traits(fw_name, adapter)

      # 2. Load Test Templates (for gen-tests)
      try:
        test_cfg = adapter.test_config
        if test_cfg:
          self.mgr.test_templates[fw_name] = test_cfg
      except AttributeError:
        pass

      # 3. Load Specifications (Hub)
      self._load_adapter_specs(adapter)

      # 4. Load Mappings (Spoke)
      self._load_adapter_definitions(fw_name, adapter)

      # 5. Load Import Namespaces
      self._load_import_namespaces(fw_name, adapter)

      # 6. Apply Dynamic Wiring logic
      self._apply_wiring(fw_name, adapter)

  @staticmethod
  def _load_trait_models(
    fw_name: str,
    adapter: FrameworkAdapter,
    config: dict[str, JsonValue],
    typed_fw: FrameworkTraits | None = None,
  ) -> None:
    """Attach structural and plugin Pydantic trait models from ``adapter`` onto ``typed_fw`` and ``config``.

    Args:
        fw_name: Framework identifier for warning diagnostics.
        adapter: Active framework adapter instance.
        config: Mutable framework configuration dictionary.
        typed_fw: Optional ``FrameworkTraits`` model updated directly without dict roundtrip.
    """
    try:
      traits = adapter.structural_traits
      if traits is not None:
        config["traits"] = traits.model_dump(exclude_unset=True)
        if typed_fw is not None and isinstance(traits, StructuralTraits):
          typed_fw.traits = traits
    except AttributeError:
      pass
    except (TypeError, ValueError) as e:
      logger.warning("⚠️ Failed to load structural traits for %s: %s", fw_name, e)

    try:
      ptraits = adapter.plugin_traits
      if ptraits is not None:
        config["plugin_traits"] = ptraits.model_dump(exclude_unset=True)
        if typed_fw is not None and isinstance(ptraits, PluginTraits):
          typed_fw.plugin_traits = ptraits
    except AttributeError:
      pass
    except (TypeError, ValueError) as e:
      logger.warning("⚠️ Failed to load plugin traits for %s: %s", fw_name, e)

  def _ensure_typed_framework_traits(self, fw_name: str) -> FrameworkTraits | None:
    """Return or create the ``FrameworkTraits`` entry for ``fw_name`` on ``self.mgr`` if supported.

    Args:
        fw_name: Framework identifier key.

    Returns:
        ``FrameworkTraits`` model stored in ``self.mgr.framework_traits``, or ``None``.
    """
    store = self._typed_store()
    if store is None:
      return None
    existing = store.framework_traits.get(fw_name)
    if existing is None:
      existing = FrameworkTraits()
      store.framework_traits[fw_name] = existing
    return existing

  def _load_adapter_traits(self, fw_name: str, adapter: FrameworkAdapter) -> None:
    """Extract import aliases, supported tiers, and structural traits.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    config = self.mgr.framework_configs[fw_name]
    typed_fw = self._ensure_typed_framework_traits(fw_name)

    try:
      imp_alias = adapter.import_alias
    except AttributeError:
      imp_alias = None
    if imp_alias:
      mod_path, alias_name = imp_alias
      config["alias"] = {"module": mod_path, "name": alias_name}
      if typed_fw is not None:
        typed_fw.alias = FrameworkAlias(module=mod_path, name=alias_name)

    try:
      inh = adapter.inherits_from
    except AttributeError:
      inh = None
    if inh:
      config["extends"] = inh
      if typed_fw is not None:
        typed_fw.extends = inh

    self._load_trait_models(fw_name, adapter, config, typed_fw=typed_fw)

    try:
      tiers = adapter.supported_tiers
    except AttributeError:
      tiers = []
    if tiers:
      config["tiers"] = [t.value for t in tiers]
      if typed_fw is not None:
        typed_fw.tiers = list(tiers)

    try:
      seed_methods = adapter.rng_seed_methods
    except AttributeError:
      seed_methods = []
    if seed_methods:
      self.mgr._known_rng_methods.update(seed_methods)

    try:
      magic_args = adapter.declared_magic_args
    except AttributeError:
      magic_args = []
    if magic_args:
      self.mgr.known_magic_args.update(magic_args)

  def _load_adapter_specs(self, adapter: FrameworkAdapter) -> None:
    """Load abstract operations defined by the adapter.

    Args:
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    try:
      specs = adapter.specifications
    except AttributeError:
      specs = {}
    if specs:
      tier = SemanticTier.EXTRAS
      for op_key in specs:
        if op_key[0].isupper():
          tier = SemanticTier.NEURAL

      store = self._typed_store()
      if store is not None:
        typed_spec_content: dict[str, JsonValueLike] = dict(specs)
        merge_tier_with_models(
          typed_store=store,
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          new_content=typed_spec_content,
          tier=tier,
        )
      else:
        spec_content: dict[str, JsonValue] = {
          op_key: op_model.model_dump(by_alias=True, exclude_unset=True) for op_key, op_model in specs.items()
        }
        merge_tier_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          new_content=spec_content,
          tier=tier,
        )

  def _load_adapter_definitions(self, fw_name: str, adapter: FrameworkAdapter) -> None:
    """Load concrete implementations defined by the adapter.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    try:
      defs = adapter.definitions
    except AttributeError:
      defs = {}
    if defs:
      for op_key in defs:
        if op_key not in self.mgr._key_origins:
          if op_key and op_key[0].isupper():
            self.mgr._key_origins[op_key] = SemanticTier.NEURAL.value
          else:
            self.mgr._key_origins[op_key] = SemanticTier.ARRAY_API.value

      store = self._typed_store()
      if store is not None:
        typed_mappings: dict[str, RuntimeTensorOrValue] = dict(defs)
        virtual_typed_snap: dict[str, RuntimeTensorOrValue] = {"__framework__": fw_name, "mappings": typed_mappings}
        merge_overlay_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          test_templates=self.mgr.test_templates,
          content=virtual_typed_snap,
          filename=f"{fw_name}_code_defs",
          typed_store=store,
        )
      else:
        mappings: dict[str, JsonValue] = {k: v.model_dump(exclude_unset=True) for k, v in defs.items()}
        virtual_snap: dict[str, JsonValue] = {"__framework__": fw_name, "mappings": mappings}
        merge_overlay_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          test_templates=self.mgr.test_templates,
          content=virtual_snap,
          filename=f"{fw_name}_code_defs",
        )

  def _load_import_namespaces(self, fw_name: str, adapter: FrameworkAdapter) -> None:
    """Register framework namespaces for import abstraction.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    try:
      ns_map = adapter.import_namespaces
    except AttributeError:
      return
    if not ns_map:
      return

    for path, config_obj in ns_map.items():
      tier = None
      alias = None

      # Only accept structured ImportConfig objects
      if isinstance(config_obj, ImportConfig):
        tier = config_obj.tier
        alias = config_obj.recommended_alias

      if tier:
        # 1. Register PROVIDER capabilities
        if fw_name not in self.mgr._providers:
          self.mgr._providers[fw_name] = {}

        root: str | None = path
        sub: str | None = None

        if alias and "." in path:
          parts = path.rsplit(".", 1)
          if len(parts) == 2 and alias == parts[1]:
            root = parts[0]
            sub = parts[1]

        provider_entry: dict[str, str | None] = {
          "root": root,
          "sub": sub,
          "alias": alias,
        }
        self.mgr._providers[fw_name][tier] = provider_entry

        # 2. Register SOURCE identification
        self.mgr._source_registry[path] = (fw_name, tier)

  def _apply_wiring(self, fw_name: str, adapter: FrameworkAdapter) -> None:
    """Execute manual wiring callback on the adapter.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    try:
      wiring_fn = adapter.apply_wiring
    except AttributeError:
      return
    dummy_snap: dict[str, dict[str, JsonValue]] = {}
    try:
      wiring_fn(dummy_snap)
      overlay_snap: dict[str, JsonValue] = {"__framework__": fw_name, **dummy_snap}
      merge_overlay_data(
        data=self.mgr.data,
        key_origins=self.mgr._key_origins,
        framework_configs=self.mgr.framework_configs,
        test_templates=self.mgr.test_templates,
        content=overlay_snap,
        filename=f"{fw_name}_dynamic_wiring",
        typed_store=self._typed_store(),
      )
    except (OSError, KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
      logger.warning("⚠️ Failed to apply wiring for %s: %s", fw_name, e)

  def _hydrate_plugins(self) -> None:
    """Load definitions from plugins that utilize auto-wire metadata.

    Returns:
        None
    """
    plugin_metadata = hooks.get_all_hook_metadata()
    store = self._typed_store()
    for spec in plugin_metadata.values():
      ops_dict: dict[str, JsonValue] = spec.ops or {}
      for op_name, op_details in ops_dict.items():
        if store is not None:
          merge_tier_with_models(
            typed_store=store,
            data=self.mgr.data,
            key_origins=self.mgr._key_origins,
            framework_configs=self.mgr.framework_configs,
            new_content={op_name: op_details},
            tier=SemanticTier.EXTRAS,
          )
        else:
          merge_tier_data(
            data=self.mgr.data,
            key_origins=self.mgr._key_origins,
            framework_configs=self.mgr.framework_configs,
            new_content={op_name: op_details},
            tier=SemanticTier.EXTRAS,
          )
