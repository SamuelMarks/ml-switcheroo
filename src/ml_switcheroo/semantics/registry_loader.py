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

from ml_switcheroo.core import hooks as hooks
from ml_switcheroo.frameworks.base import FrameworkAdapter, ImportConfig, available_frameworks, get_adapter
from ml_switcheroo.semantics.merging import merge_overlay_data, merge_tier_data

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

  def index_variants(self) -> None:
    """Index operation variants into hash maps for O(1) lookups.

    Returns:
        None
    """
    if hasattr(self.mgr, "_variant_cache"):
      self.mgr._variant_cache.clear()
      for abstract_id, details in self.mgr.data.items():
        if isinstance(details, dict):
          variants = details.get("variants", {})
          if isinstance(variants, dict):
            for fw_name, variant in variants.items():
              if isinstance(variant, dict):
                self.mgr._variant_cache[(abstract_id, fw_name)] = variant

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
      if hasattr(adapter, "test_config") and adapter.test_config:
        self.mgr.test_templates[fw_name] = adapter.test_config

      # 3. Load Specifications (Hub)
      self._load_adapter_specs(adapter)

      # 4. Load Mappings (Spoke)
      self._load_adapter_definitions(fw_name, adapter)

      # 5. Load Import Namespaces
      self._load_import_namespaces(fw_name, adapter)

      # 6. Apply Dynamic Wiring logic
      self._apply_wiring(fw_name, adapter)

  def _load_adapter_traits(
    self, fw_name: str, adapter: typing.Union[FrameworkAdapter, typing.Union[typing.Hashable, typing.Sized]]
  ) -> None:
    """Extract import aliases, supported tiers, and structural traits.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    config = self.mgr.framework_configs[fw_name]

    if hasattr(adapter, "import_alias") and getattr(adapter, "import_alias"):
      mod_path, alias_name = getattr(adapter, "import_alias")
      config["alias"] = {
        "module": mod_path,
        "name": alias_name,
      }

    if hasattr(adapter, "inherits_from") and getattr(adapter, "inherits_from"):
      config["extends"] = getattr(adapter, "inherits_from")

    if hasattr(adapter, "structural_traits"):
      try:
        traits = getattr(adapter, "structural_traits")
        if traits:
          config["traits"] = traits.model_dump(exclude_unset=True)
      except (AttributeError, TypeError, ValueError) as e:
        logger.warning("⚠️ Failed to load structural traits for %s: %s", fw_name, e)

    if hasattr(adapter, "plugin_traits"):
      try:
        ptraits = getattr(adapter, "plugin_traits")
        if ptraits:
          config["plugin_traits"] = ptraits.model_dump(exclude_unset=True)
      except (AttributeError, TypeError, ValueError) as e:
        logger.warning("⚠️ Failed to load plugin traits for %s: %s", fw_name, e)

    if hasattr(adapter, "supported_tiers") and getattr(adapter, "supported_tiers"):
      config["tiers"] = [t.value for t in getattr(adapter, "supported_tiers")]

    if hasattr(adapter, "rng_seed_methods") and getattr(adapter, "rng_seed_methods"):
      self.mgr._known_rng_methods.update(getattr(adapter, "rng_seed_methods"))

    if hasattr(adapter, "declared_magic_args") and getattr(adapter, "declared_magic_args"):
      self.mgr.known_magic_args.update(getattr(adapter, "declared_magic_args"))

  def _load_adapter_specs(
    self, adapter: typing.Union[FrameworkAdapter, typing.Union[typing.Hashable, typing.Sized]]
  ) -> None:
    """Load abstract operations defined by the adapter.

    Args:
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    if hasattr(adapter, "specifications") and getattr(adapter, "specifications"):
      specs = getattr(adapter, "specifications")
      # Default to Extras unless inferred otherwise
      tier = SemanticTier.EXTRAS
      spec_content: typing.Dict[str, JsonValue] = {}

      for op_key, op_model in specs.items():
        spec_content[op_key] = op_model.model_dump(by_alias=True, exclude_unset=True)
        if op_key[0].isupper():
          tier = SemanticTier.NEURAL

      merge_tier_data(
        data=self.mgr.data,
        key_origins=self.mgr._key_origins,
        framework_configs=self.mgr.framework_configs,
        new_content=spec_content,
        tier=tier,
      )

  def _load_adapter_definitions(
    self, fw_name: str, adapter: typing.Union[FrameworkAdapter, typing.Union[typing.Hashable, typing.Sized]]
  ) -> None:
    """Load concrete implementations defined by the adapter.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    if hasattr(adapter, "definitions") and getattr(adapter, "definitions"):
      defs = getattr(adapter, "definitions")
      mappings: typing.Dict[str, JsonValue] = {k: v.model_dump(exclude_unset=True) for k, v in defs.items()}

      # Pre-label Tiers based on Naming Convention if unknown
      for op_key in mappings.keys():
        if op_key not in self.mgr._key_origins:
          if op_key and op_key[0].isupper():
            self.mgr._key_origins[op_key] = SemanticTier.NEURAL.value
          else:
            self.mgr._key_origins[op_key] = SemanticTier.ARRAY_API.value

      virtual_snap: typing.Dict[str, JsonValue] = {"__framework__": fw_name, "mappings": mappings}
      merge_overlay_data(
        data=self.mgr.data,
        key_origins=self.mgr._key_origins,
        framework_configs=self.mgr.framework_configs,
        test_templates=self.mgr.test_templates,
        content=virtual_snap,
        filename=f"{fw_name}_code_defs",
      )

  def _load_import_namespaces(
    self, fw_name: str, adapter: typing.Union[FrameworkAdapter, typing.Union[typing.Hashable, typing.Sized]]
  ) -> None:
    """Register framework namespaces for import abstraction.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    if not hasattr(adapter, "import_namespaces") or not getattr(adapter, "import_namespaces"):
      return

    for path, config_obj in getattr(adapter, "import_namespaces").items():
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

        root: typing.Optional[str] = path
        sub: typing.Optional[str] = None

        if alias and "." in path:
          parts = path.rsplit(".", 1)
          if len(parts) == 2 and alias == parts[1]:
            root = parts[0]
            sub = parts[1]

        provider_entry: typing.Dict[str, typing.Optional[str]] = {
          "root": root,
          "sub": sub,
          "alias": alias,
        }
        self.mgr._providers[fw_name][tier] = provider_entry

        # 2. Register SOURCE identification
        self.mgr._source_registry[path] = (fw_name, tier)

  def _apply_wiring(
    self, fw_name: str, adapter: typing.Union[FrameworkAdapter, typing.Union[typing.Hashable, typing.Sized]]
  ) -> None:
    """Execute manual wiring callback on the adapter.

    Args:
        fw_name: The name of the framework.
        adapter: The active framework adapter instance.

    Returns:
        None
    """
    if hasattr(adapter, "apply_wiring"):
      dummy_snap: typing.Dict[str, typing.Dict[str, JsonValue]] = {"__framework__": fw_name}  # type: ignore[dict-item]
      try:
        getattr(adapter, "apply_wiring")(dummy_snap)
        merge_overlay_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          test_templates=self.mgr.test_templates,
          content=dummy_snap,
          filename=f"{fw_name}_dynamic_wiring",
        )
      except (OSError, KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
        logger.warning("⚠️ Failed to apply wiring for %s: %s", fw_name, e)

  def _hydrate_plugins(self) -> None:
    """Load definitions from plugins that utilize auto-wire metadata.

    Returns:
        None
    """
    plugin_metadata = hooks.get_all_hook_metadata()
    for _, spec in plugin_metadata.items():
      ops_dict: typing.Dict[str, JsonValue] = getattr(spec, "ops", None) or {}
      for op_name, op_details in ops_dict.items():
        merge_tier_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          framework_configs=self.mgr.framework_configs,
          new_content={op_name: op_details},
          tier=SemanticTier.EXTRAS,
        )
