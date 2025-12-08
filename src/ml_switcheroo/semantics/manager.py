"""
Semantics Manager for Knowledge Base Loading and Updating.

This module is responsible for locating, loading, and merging semantic
specification files (JSONs) into a unified Knowledge Graph.
It integrates with `testing.registry_sync` to auto-discover test templates.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from pydantic import ValidationError

if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  files = None

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.testing.registry_sync import TemplateGenerator
from ml_switcheroo.semantics.schema import OpDefinition


def resolve_semantics_dir() -> Path:
  if sys.version_info >= (3, 9) and files:
    return Path(str(files("ml_switcheroo.semantics")))
  return Path(__file__).parent


# Standard Fallback Templates (Bootstrap)
# Note: These are now also defined in k_test_templates.json.
# We keep them here for safe fallback if the JSON is missing or corrupt.
_DEFAULT_TEMPLATES = {
  "torch": {
    "import": "import torch",
    "convert_input": "torch.from_numpy({np_var})",
    "to_numpy": "{res_var}.detach().cpu().numpy() if hasattr({res_var}, 'detach') else {res_var}",
  },
  "jax": {
    "import": "import jax\nimport jax.numpy as jnp",
    "convert_input": "jnp.array({np_var})",
    "to_numpy": "np.array({res_var})",
    "jit_wrap": "True",
  },
  "tensorflow": {
    "import": "import tensorflow as tf",
    "convert_input": "tf.convert_to_tensor({np_var})",
    "to_numpy": "{res_var}.numpy()",
  },
  "mlx": {
    "import": "import mlx.core as mx",
    "convert_input": "mx.array({np_var})",
    "to_numpy": "np.array({res_var})",
  },
  "numpy": {
    "import": "import numpy as np",
    "convert_input": "{np_var}",
    "to_numpy": "{res_var}",
  },
}

# Default import conventions (if not overridden by JSON)
_DEFAULT_ALIASES = {
  "jax": {"module": "jax.numpy", "alias": "jnp"},
  "tensorflow": {"module": "tensorflow", "alias": "tf"},
  "mlx": {"module": "mlx.core", "alias": "mx"},
  "numpy": {"module": "numpy", "alias": "np"},
}


class SemanticsManager:
  """
  Central database for semantic mappings and configuration.
  """

  def __init__(self):
    self.data: Dict[str, Dict] = {}
    self.import_data: Dict[str, Dict] = {}
    self.framework_configs: Dict[str, Dict] = {}

    # 1. Initialize with hardcoded defaults
    self.test_templates: Dict[str, Dict] = _DEFAULT_TEMPLATES.copy()

    self._reverse_index: Dict[str, Tuple[str, Dict]] = {}
    self._key_origins: Dict[str, str] = {}
    self._validation_status: Dict[str, bool] = {}

    # 2. Sync code-defined logic (Registry Adapters)
    # We do this BEFORE loading JSON so the file on disk acts as the final override.
    code_templates = TemplateGenerator.generate_templates(self.test_templates)
    for fw, tpl in code_templates.items():
      if fw in self.test_templates:
        self.test_templates[fw].update(tpl)
      else:
        self.test_templates[fw] = tpl

    # 3. Load Knowledge Graph (JSON files)
    # This merges k_test_templates.json on top of the code/defaults.
    self._load_knowledge_graph()

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

  def update_definition(self, abstract_id: str, new_data: Dict[str, Any]) -> None:
    # Attempt to validate update before applying
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

        file_content[abstract_id] = final_data

        with open(file_path, "w", encoding="utf-8") as f:
          json.dump(file_content, f, indent=2, sort_keys=True)

      except Exception as e:
        print(f"❌ Failed to write update for {abstract_id} to {filename}: {e}")

  def _load_knowledge_graph(self) -> None:
    base_path = resolve_semantics_dir()

    load_order = [
      (SemanticTier.ARRAY_API, "k_array_api.json"),
      (SemanticTier.NEURAL, "k_neural_net.json"),
      (SemanticTier.EXTRAS, "k_framework_extras.json"),
    ]

    files_found = 0
    for tier, filename in load_order:
      full_path = base_path / filename
      if full_path.exists():
        files_found += 1
        try:
          with open(full_path, "r", encoding="utf-8") as f:
            content = json.load(f)
          self._merge_tier(content, tier)
        except json.JSONDecodeError as e:
          print(f"❌ Error decoding {filename}: {e}")

    templates_path = base_path / "k_test_templates.json"
    if templates_path.exists():
      try:
        with open(templates_path, "r", encoding="utf-8") as f:
          content = json.load(f)
        self._merge_templates(content)
      except json.JSONDecodeError as e:
        print(f"❌ Error decoding {templates_path.name}: {e}")

    self._build_index()

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
            f"Conflict detected for '{op_name}': Defined in '{prev_tier}' but overwritten by '{tier}'.",
            UserWarning,
          )

      # Feature 093: Schema Validation
      try:
        validated_op = OpDefinition.model_validate(details)
        # Convert back to dict for internal storage to maintain dict-access API compatibility
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
        self.framework_configs[fw_name].update(traits)

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
    # Check loaded configuration first (JSON or Registry updated)
    if framework in self.test_templates:
      return self.test_templates[framework]

    # Fallback to Hardcoded Defaults (important for tests verifying fallback)
    return _DEFAULT_TEMPLATES.get(framework)

  def get_framework_aliases(self) -> Dict[str, Tuple[str, str]]:
    # Start with defaults
    result: Dict[str, Tuple[str, str]] = {}
    for fw, conf in _DEFAULT_ALIASES.items():
      result[fw] = (conf["module"], conf["alias"])

    # Override with loaded configs
    config_source = getattr(self, "framework_configs", {})
    for fw, config in config_source.items():
      alias_conf = config.get("alias")
      if alias_conf and isinstance(alias_conf, dict):
        mod = alias_conf.get("module")
        alias = alias_conf.get("name")
        if mod and alias:
          result[fw] = (mod, alias)

    return result
