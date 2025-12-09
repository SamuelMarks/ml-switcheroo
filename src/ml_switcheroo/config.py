import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
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

  Logic:
  1. Retrieve registered frameworks.
  2. Instantiate adapter to read 'ui_priority'.
  3. Sort by priority (asc), then by name (asc) for stability.
  4. Frameworks without adapters sort to the end (priority 999).
  """
  frameworks = available_frameworks()

  def sort_key(name: str):
    adapter = get_adapter(name)
    priority = 999
    if adapter and hasattr(adapter, "ui_priority"):
      try:
        priority = int(adapter.ui_priority)
      except (ValueError, TypeError):
        pass
    return (priority, name)

  return sorted(frameworks, key=sort_key)


class RuntimeConfig(BaseModel):
  """
  Global configuration container for the translation engine.
  """

  # Accept string inputs, validated dynamically against the registry
  source_framework: str = "torch"
  target_framework: str = "jax"
  strict_mode: bool = False
  plugin_settings: Dict[str, Any] = Field(default_factory=dict)
  plugin_paths: List[Path] = Field(default_factory=list)
  validation_report: Optional[Path] = None

  @field_validator("source_framework", "target_framework")
  @classmethod
  def validate_framework(cls, v: str) -> str:
    """Ensures the framework is registered in the system."""
    # Normalize input
    v_clean = v.lower().strip()

    # Allow bypassing if no frameworks loaded (bootstrap/test enviroments)
    known = available_frameworks()
    if known and v_clean not in known:
      raise ValueError(
        f"Unknown framework '{v}'. Available: {known}. Add {v}.py to src/ml_switcheroo/frameworks/ to enable support."
      )
    return v_clean

  def parse_plugin_settings(self, schema: Type[T]) -> T:
    try:
      return schema.model_validate(self.plugin_settings)
    except ValidationError as e:
      raise ValueError(f"Plugin configuration validation failed: {e}")

  @classmethod
  def load(
    cls,
    source: Optional[str] = None,
    target: Optional[str] = None,
    strict_mode: Optional[bool] = None,
    plugin_settings: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Path] = None,
    search_path: Optional[Path] = None,
  ) -> "RuntimeConfig":
    start_dir = search_path or Path.cwd()
    toml_config, toml_dir = _load_toml_settings(start_dir)

    # 1. Framework Defaults
    final_source = source or toml_config.get("source_framework", "torch")
    final_target = target or toml_config.get("target_framework", "jax")

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
      strict_mode=final_strict,
      plugin_settings=final_plugins,
      plugin_paths=final_plugin_paths,
      validation_report=final_report,
    )


def _load_toml_settings(start_path: Path) -> tuple[Dict[str, Any], Optional[Path]]:
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
