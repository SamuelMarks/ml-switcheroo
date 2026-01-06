"""
Registry Hydration Logic.

This module introspects the active Python environment to extract semantic definitions
from registered Framework Adapters and Plugin Hooks. This enables "Code-First"
definitions alongside "Config-First" JSONs.
"""

from typing import Any
from ml_switcheroo.frameworks.base import (
  available_frameworks,
  get_adapter,
  ImportConfig,
)
from ml_switcheroo.core import hooks
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.merging import merge_tier_data, merge_overlay_data


class RegistryLoader:
  """
  Hydrates the SemanticsManager from python objects (Adapters/Plugins).
  """

  def __init__(self, manager: Any):
    """
    Initialize the loader.

    Args:
        manager: The parent SemanticsManager instance.
    """
    self.mgr = manager

  def hydrate(self) -> None:
    """
    Main entry point. Scans adapters and plugins.
    """
    self._hydrate_adapters()
    self._hydrate_plugins()

  def _hydrate_adapters(self) -> None:
    """
    Iterates over registered FrameworkAdapters to extract Traits and Mappings.
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

  def _load_adapter_traits(self, fw_name: str, adapter: Any) -> None:
    """Extracts import aliases, supported tiers, and structural traits."""
    config = self.mgr.framework_configs[fw_name]

    if hasattr(adapter, "import_alias") and adapter.import_alias:
      mod_path, alias_name = adapter.import_alias
      config["alias"] = {
        "module": mod_path,
        "name": alias_name,
      }

    if hasattr(adapter, "structural_traits"):
      try:
        traits = adapter.structural_traits
        if traits:
          config["traits"] = traits.model_dump(exclude_unset=True)
      except Exception as e:
        print(f"⚠️ Failed to load structural traits for {fw_name}: {e}")

    if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:
      config["tiers"] = [t.value for t in adapter.supported_tiers]

    if hasattr(adapter, "rng_seed_methods") and adapter.rng_seed_methods:
      self.mgr._known_rng_methods.update(adapter.rng_seed_methods)

    if hasattr(adapter, "declared_magic_args") and adapter.declared_magic_args:
      self.mgr.known_magic_args.update(adapter.declared_magic_args)

  def _load_adapter_specs(self, adapter: Any) -> None:
    """Loads abstract operations defined by the adapter."""
    if hasattr(adapter, "specifications") and adapter.specifications:
      specs = adapter.specifications
      # Default to Extras unless inferred otherwise
      tier = SemanticTier.EXTRAS
      spec_content = {}

      for op_key, op_model in specs.items():
        spec_content[op_key] = op_model.model_dump(by_alias=True, exclude_unset=True)
        if op_key[0].isupper():
          tier = SemanticTier.NEURAL

      merge_tier_data(
        data=self.mgr.data,
        key_origins=self.mgr._key_origins,
        import_data={},
        framework_configs=self.mgr.framework_configs,
        new_content=spec_content,
        tier=tier,
      )

  def _load_adapter_definitions(self, fw_name: str, adapter: Any) -> None:
    """Loads concrete implementations defined by the adapter."""
    if hasattr(adapter, "definitions") and adapter.definitions:
      defs = adapter.definitions
      mappings = {k: v.model_dump(exclude_unset=True) for k, v in defs.items()}

      # Pre-label Tiers based on Naming Convention if unknown
      for op_key in mappings.keys():
        if op_key not in self.mgr._key_origins:
          if op_key and op_key[0].isupper():
            self.mgr._key_origins[op_key] = SemanticTier.NEURAL.value
          else:
            self.mgr._key_origins[op_key] = SemanticTier.ARRAY_API.value

      virtual_snap = {"__framework__": fw_name, "mappings": mappings}
      merge_overlay_data(
        data=self.mgr.data,
        key_origins=self.mgr._key_origins,
        import_data={},
        framework_configs=self.mgr.framework_configs,
        test_templates=self.mgr.test_templates,
        content=virtual_snap,
        filename=f"{fw_name}_code_defs",
      )

  def _load_import_namespaces(self, fw_name: str, adapter: Any) -> None:
    """Registers framework namespaces for import abstraction."""
    if not hasattr(adapter, "import_namespaces") or not adapter.import_namespaces:
      return

    for path, config_obj in adapter.import_namespaces.items():
      tier = None
      alias = None

      if isinstance(config_obj, ImportConfig):
        tier = config_obj.tier
        alias = config_obj.recommended_alias
      elif isinstance(config_obj, dict):
        # Legacy format fallback
        tier = SemanticTier.EXTRAS
        alias = config_obj.get("alias")

      if tier:
        # 1. Register PROVIDER capabilities
        if fw_name not in self.mgr._providers:
          self.mgr._providers[fw_name] = {}

        root = path
        sub = None

        if alias and "." in path:
          parts = path.rsplit(".", 1)
          if len(parts) == 2 and alias == parts[1]:
            root = parts[0]
            sub = parts[1]

        self.mgr._providers[fw_name][tier] = {
          "root": root,
          "sub": sub,
          "alias": alias,
        }

        # 2. Register SOURCE identification
        self.mgr._source_registry[path] = (fw_name, tier)

  def _apply_wiring(self, fw_name: str, adapter: Any) -> None:
    """Executes manual wiring callback on the adapter."""
    if hasattr(adapter, "apply_wiring"):
      dummy_snap = {"__framework__": fw_name}
      try:
        adapter.apply_wiring(dummy_snap)
        merge_overlay_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          import_data={},
          framework_configs=self.mgr.framework_configs,
          test_templates=self.mgr.test_templates,
          content=dummy_snap,
          filename=f"{fw_name}_dynamic_wiring",
        )
      except Exception as e:
        print(f"⚠️ Failed to apply wiring for {fw_name}: {e}")

  def _hydrate_plugins(self) -> None:
    """
    Loads definitions from plugins that utilize auto-wire metadata.
    """
    plugin_metadata = hooks.get_all_hook_metadata()
    for _, spec in plugin_metadata.items():
      for op_name, op_details in spec.ops.items():
        merge_tier_data(
          data=self.mgr.data,
          key_origins=self.mgr._key_origins,
          import_data={},
          framework_configs=self.mgr.framework_configs,
          new_content={op_name: op_details},
          tier=SemanticTier.EXTRAS,
        )
