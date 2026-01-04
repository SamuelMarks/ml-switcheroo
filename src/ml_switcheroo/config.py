"""
Runtime Configuration Store.
Supports Dynamic Defaults, TOML loading, and Framework Flavours.

This module resolves default Source and Target frameworks by querying the
``ui_priority`` of registered adapters, ensuring the logic is completely agnostic
to the specific libraries installed.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
from pydantic import BaseModel, Field, ValidationError, field_validator

from ml_switcheroo.frameworks import available_frameworks, get_adapter

# Optional TOML support
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

  The sort order is determined by the ``ui_priority`` attribute of the
  registered FrameworkAdapter. Lower numbers appear first (Source default).

  If no priority is defined, defaults to 999.

  Returns:
      List[str]: Sorted list of framework identifiers (e.g. ['torch', 'jax']).
  """
  frameworks = available_frameworks()

  def sort_key(name: str) -> Tuple[int, str]:
    adapter = get_adapter(name)
    priority = 999
    is_child = False

    if adapter:
      # Check hierarchy to push flavours/children to the end
      if hasattr(adapter, "inherits_from") and adapter.inherits_from:
        is_child = True

      # Extract priority
      if hasattr(adapter, "ui_priority"):
        try:
          priority = int(adapter.ui_priority)
        except (ValueError, TypeError):
          pass

    # Move children (flavours) to the very end for UI clarity
    if is_child:
      priority = 9999

    # Tuple sort: Priority first, then Alphabetical tie-break
    return (priority, name)

  return sorted(frameworks, key=sort_key)


def _resolve_default_source() -> str:
  """
  Resolves the default source framework.

  Returns:
      str: The highest priority framework key (Index 0 in sorted list),
      or "source_placeholder" if none are registered.
  """
  fws = get_framework_priority_order()
  return fws[0] if fws else "source_placeholder"


def _resolve_default_target() -> str:
  """
  Resolves the default target framework.

  Returns:
      str: The second highest priority framework key (Index 1),
      or the first (Index 0) if only one exists, or "target_placeholder".
  """
  fws = get_framework_priority_order()
  if len(fws) >= 2:
    return fws[1]
  return fws[0] if fws else "target_placeholder"


class RuntimeConfig(BaseModel):
  """
  Global configuration container for the translation engine.
  """

  source_framework: str = Field(
    default_factory=_resolve_default_source,
    description="The primary source framework key.",
  )
  target_framework: str = Field(
    default_factory=_resolve_default_target,
    description="The primary target framework key.",
  )

  # Framework Flavours (e.g. 'flax_nnx' when main is 'jax')
  source_flavour: Optional[str] = Field(None, description="Detailed source sub-framework.")
  target_flavour: Optional[str] = Field(None, description="Detailed target sub-framework.")

  strict_mode: bool = Field(False, description="If True, fail on unknown APIs.")

  # Structural Verification
  intermediate: Optional[str] = Field(
    None, description="Intermediate representation layer (e.g. 'mlir', 'tikz') for round-trip verification."
  )

  enable_fusion: bool = Field(False, description="If True, performs graph-level optimization and fusion.")

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
        str: The normalized framework key.

    Raises:
        ValueError: If the framework is not registered and not a placeholder.
    """
    v_clean = v.lower().strip()
    known = available_frameworks()
    # Allow unregistered checks if registry is empty (bootstrap/test mode)
    # or if using the placeholders generated when empty.
    if known and v_clean not in known and v_clean not in ["source_placeholder", "target_placeholder"]:
      raise ValueError(f"Unknown framework: '{v_clean}'. Supported frameworks: {known}")
    return v_clean

  @property
  def effective_source(self) -> str:
    """
    Returns the resolved source framework key.

    Prioritizes ``source_flavour`` if present, otherwise returns ``source_framework``.

    Returns:
        str: The active source framework key.
    """
    return self.source_flavour if self.source_flavour else self.source_framework

  @property
  def effective_target(self) -> str:
    """
    Returns the resolved target framework key.

    Prioritizes ``target_flavour`` if present, otherwise returns ``target_framework``.

    Returns:
        str: The active target framework key.
    """
    return self.target_flavour if self.target_flavour else self.target_framework

  def parse_plugin_settings(self, schema: Type[T]) -> T:
    """
    Validates plugin settings against a Pydantic model.

    Args:
        schema (Type[T]): The Pydantic model class defining the settings schema.

    Returns:
        T: An instance of the schema populated with validated settings.

    Raises:
        ValueError: If validation against the schema fails.
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
    intermediate: Optional[str] = None,
    enable_fusion: Optional[bool] = None,
    plugin_settings: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Path] = None,
    search_path: Optional[Path] = None,
  ) -> "RuntimeConfig":
    """
    Loads configuration from ``pyproject.toml``, overriding with CLI arguments.
    Defaults are calculated dynamically via factory methods if not found.

    Args:
        source (Optional[str]): Override source framework.
        target (Optional[str]): Override target framework.
        source_flavour (Optional[str]): Override source flavour.
        target_flavour (Optional[str]): Override target flavour.
        strict_mode (Optional[bool]): Override strict mode setting.
        intermediate (Optional[str]): Override intermediate representation mode.
        enable_fusion (Optional[bool]): Override fusion optimization setting.
        plugin_settings (Optional[Dict]): Additional plugin settings.
        validation_report (Optional[Path]): Override validation report path.
        search_path (Optional[Path]): Directory path to search for config file.

    Returns:
        RuntimeConfig: The fully resolved configuration object.
    """
    start_dir = search_path or Path.cwd()
    toml_config, toml_dir = _load_toml_settings(start_dir)

    # 1. Framework Defaults
    final_source = source if source else toml_config.get("source_framework", _resolve_default_source())
    final_target = target if target else toml_config.get("target_framework", _resolve_default_target())

    # 1b. Flavours
    final_src_flavour = source_flavour or toml_config.get("source_flavour")
    final_tgt_flavour = target_flavour or toml_config.get("target_flavour")

    # 2. Strict Mode
    if strict_mode is not None:
      final_strict = strict_mode
    else:
      final_strict = toml_config.get("strict_mode", False)

    # 2b. Intermediate Mode
    final_intermediate = intermediate or toml_config.get("intermediate")

    # 2c. Fusion
    if enable_fusion is not None:
      final_fusion = enable_fusion
    else:
      final_fusion = toml_config.get("enable_fusion", False)

    # 3. Plugin Settings (Merge TOML + CLI, CLI wins)
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
      intermediate=final_intermediate,
      enable_fusion=final_fusion,
      plugin_settings=final_plugins,
      plugin_paths=final_plugin_paths,
      validation_report=final_report,
    )


def _load_toml_settings(start_path: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
  """
  Recursively searches parents for 'pyproject.toml' and extracts config.

  Args:
      start_path (Path): Directory to begin the search from.

  Returns:
      Tuple[Dict[str, Any], Optional[Path]]:
          - A dictionary of configuration settings found in the TOML file.
          - The directory containing the TOML file, or None if not found.
  """
  if not tomllib:
    return {}, None

  current = start_path.resolve()

  # Iterate current and parents
  for path in [current, *current.parents]:
    toml_path = path / "pyproject.toml"
    if toml_path.exists() and toml_path.is_file():
      try:
        with open(toml_path, "rb") as f:
          data = tomllib.load(f)

        tool_section = data.get("tool", {})
        return tool_section.get("ml_switcheroo", {}), path
      except Exception:
        # Malformed TOML or permission error implies skip
        return {}, None

  return {}, None


def parse_cli_key_values(items: Optional[List[str]]) -> Dict[str, Any]:
  """
  Parses a list of 'key=value' strings into a dictionary with type inference.

  Args:
      items (Optional[List[str]]): List of strings in "key=value" format.

  Returns:
      Dict[str, Any]: Dictionary with parsed values (bool, int, float, or str).
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

    # Basic type inference
    lower_val = val_str.lower()
    if lower_val == "true":
      final_val = True
    elif lower_val == "false":
      final_val = False
    else:
      try:
        if "." in val_str or "e" in lower_val:
          final_val = float(val_str)
        else:
          final_val = int(val_str)
      except ValueError:
        pass

    config[key] = final_val

  return config
