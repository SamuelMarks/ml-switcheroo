"""
Runtime Configuration Store.
Updated to support Framework Flavours (Hierarchical Selection).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator

from ml_switcheroo.frameworks import available_frameworks, get_adapter

if sys.version_info >= (3, 11):
  import tomllib
else:
  try:
    import tomli as tomllib
  except ImportError:
    tomllib = None  # type: ignore

T = TypeVar("T", bound=BaseModel)


def get_framework_priority_order() -> List[str]:
  """
  Returns a list of framework keys sorted by UI Priority.

  The sort order is determined by the `ui_priority` attribute of the
  registered FrameworkAdapter. Lower numbers appear first.

  Returns:
      List[str]: Sorted list of framework identifiers (e.g. ['torch', 'jax']).
  """
  frameworks = available_frameworks()

  def sort_key(name: str) -> Tuple[int, str]:
    adapter = get_adapter(name)
    priority = 999
    is_child = False

    if adapter:
      if hasattr(adapter, "inherits_from") and adapter.inherits_from:
        is_child = True

      if hasattr(adapter, "ui_priority"):
        try:
          priority = int(adapter.ui_priority)
        except (ValueError, TypeError):
          pass

    # Move children (flavours) to the very end or filter them out in UI logic
    if is_child:
      priority = 9999

    return (priority, name)

  return sorted(frameworks, key=sort_key)


class RuntimeConfig(BaseModel):
  """
  Global configuration container for the translation engine.
  """

  source_framework: str = Field("torch", description="The primary source framework key (e.g., 'torch').")
  target_framework: str = Field("jax", description="The primary target framework key (e.g., 'jax').")

  # Feature 06: Framework Flavours
  # Used to specify sub-frameworks (e.g. 'flax_nnx') when the main selection is generic ('jax').
  source_flavour: Optional[str] = Field(None, description="Detailed source sub-framework (e.g. 'flax_nnx').")
  target_flavour: Optional[str] = Field(None, description="Detailed target sub-framework.")

  strict_mode: bool = Field(False, description="If True, fail on unknown APIs. If False, pass through.")
  plugin_settings: Dict[str, Any] = Field(default_factory=dict, description="Configuration passed to plugins.")
  plugin_paths: List[Path] = Field(default_factory=list, description="External directories to scan for plugins.")
  validation_report: Optional[Path] = Field(None, description="Path to a verification lockfile.")

  @field_validator("source_framework", "target_framework")
  @classmethod
  def validate_framework(cls, v: str) -> str:
    """
    Ensures the framework is registered in the system.

    Args:
        v (str): The framework key to validate.

    Returns:
        str: The normalized (lowercase) framework key.

    Raises:
        ValueError: If the framework is not found in the registry.
    """
    v_clean = v.lower().strip()
    known = available_frameworks()
    # We allow unregistered frameworks if the registry is empty (bootstrap/test mode)
    if known and v_clean not in known:
      raise ValueError(f"Unknown framework: '{v_clean}'. Supported frameworks: {known}")
    return v_clean

  @property
  def effective_source(self) -> str:
    """
    Resolves the specific framework key to use for source logic.

    If a flavour (e.g. 'flax_nnx') is provided, it overrides the general framework ('jax').

    Returns:
        str: The active source framework key.
    """
    return self.source_flavour if self.source_flavour else self.source_framework

  @property
  def effective_target(self) -> str:
    """
    Resolves the specific framework key to use for target logic.

    Returns:
        str: The active target framework key.
    """
    return self.target_flavour if self.target_flavour else self.target_framework

  def parse_plugin_settings(self, schema: Type[T]) -> T:
    """
    Validates the raw plugin settings dictionary against a specific Pydantic model.

    Args:
        schema (Type[T]): The Pydantic model class defining expected settings.

    Returns:
        T: An instance of the schema model populated with runtime values.
    """
    try:
      return schema.model_validate(self.plugin_settings)
    except ValidationError as e:
      raise ValueError(f"Plugin configuration validation failed: {e}")

  @classmethod
  def load(
    cls,
    source: Optional[str] = None,
    target: Optional[str] = None,
    source_flavour: Optional[str] = None,
    target_flavour: Optional[str] = None,
    strict_mode: Optional[bool] = None,
    plugin_settings: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Path] = None,
    search_path: Optional[Path] = None,
  ) -> "RuntimeConfig":
    """
    Loads configuration from pyproject.toml and overrides with CLI arguments.

    Args:
        source (Optional[str]): Override for source framework.
        target (Optional[str]): Override for target framework.
        source_flavour (Optional[str]): Override for source flavour.
        target_flavour (Optional[str]): Override for target flavour.
        strict_mode (Optional[bool]): Override for strict mode setting.
        plugin_settings (Optional[Dict]): Additional CLI plugin settings.
        validation_report (Optional[Path]): Override for validation report path.
        search_path (Optional[Path]): Directory to start searching for TOML config.

    Returns:
        RuntimeConfig: The fully resolved configuration object.
    """
    start_dir = search_path or Path.cwd()
    toml_config, toml_dir = _load_toml_settings(start_dir)

    # 1. Framework Defaults
    final_source = source or toml_config.get("source_framework", "torch")
    final_target = target or toml_config.get("target_framework", "jax")

    # 1b. Flavours
    final_src_flavour = source_flavour or toml_config.get("source_flavour")
    final_tgt_flavour = target_flavour or toml_config.get("target_flavour")

    # 2. Strict Mode
    if strict_mode is not None:
      final_strict = strict_mode
    else:
      final_strict = toml_config.get("strict_mode", False)

    # 3. Plugin Settings
    toml_plugins = toml_config.get("plugin_settings", {})
    cli_plugins = plugin_settings or {}
    final_plugins = {**toml_plugins, **cli_plugins}

    # 4. Validation Report
    final_report = validation_report
    if not final_report and "validation_report" in toml_config:
      final_report = Path(toml_config["validation_report"])

    # 5. External Plugins
    raw_paths = toml_config.get("plugin_paths", [])
    final_plugin_paths = []
    if toml_dir:
      for p in raw_paths:
        resolved = (toml_dir / Path(p)).resolve()
        final_plugin_paths.append(resolved)
    else:
      final_plugin_paths = [Path(p).resolve() for p in raw_paths]

    return cls(
      source_framework=final_source,
      target_framework=final_target,
      source_flavour=final_src_flavour,
      target_flavour=final_tgt_flavour,
      strict_mode=final_strict,
      plugin_settings=final_plugins,
      plugin_paths=final_plugin_paths,
      validation_report=final_report,
    )


def _load_toml_settings(start_path: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
  """
  Recursively searches parents for 'pyproject.toml' and extracts config.

  Args:
      start_path (Path): Directory to start search from.

  Returns:
      Tuple[Dict, Optional[Path]]: The config dict and the directory definition was found in.
  """
  if not tomllib:
    return {}, None

  current = start_path.resolve()

  for parent in [current, *current.parents]:
    toml_path = parent / "pyproject.toml"
    if toml_path.exists() and toml_path.is_file():
      try:
        with open(toml_path, "rb") as f:
          data = tomllib.load(f)

        tool_section = data.get("tool", {})
        return tool_section.get("ml_switcheroo", {}), parent
      except Exception:
        return {}, None

  return {}, None


def parse_cli_key_values(items: Optional[List[str]]) -> Dict[str, Any]:
  """
  Parses a list of 'key=value' strings into a dictionary.

  Types are inferred (int, float, bool, or string).

  Args:
      items (Optional[List[str]]): List of raw CLI strings directly from argparse.

  Returns:
      Dict[str, Any]: Parsed dictionary.
  """
  if not items:
    return {}

  config = {}
  for item in items:
    if "=" not in item:
      print(f"⚠️  Ignoring invalid config format: '{item}'. Expected 'key=value'.")
      continue

    key, val_str = item.split("=", 1)
    key = key.strip()
    val_str = val_str.strip()

    final_val: Any = val_str

    if val_str.lower() == "true":
      final_val = True
    elif val_str.lower() == "false":
      final_val = False
    else:
      try:
        if "." in val_str or "e" in val_str:
          final_val = float(val_str)
        else:
          final_val = int(val_str)
      except ValueError:
        pass

    config[key] = final_val

  return config
