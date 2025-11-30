"""
Plugin Registry, Hook Context, and Dynamic Loader.

This module provides the infrastructure for extending ml-switcheroo via plugins.
It enables developers to intercept and modify the Abstract Syntax Tree (AST)
during the conversion process using a hook-based system.

## How to write a Plugin

A plugin is a simple Python function decorated with `@register_hook`. It receives
the current AST node (usually a `Call`) and a context object.

### 1. Define the Hook

```python
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("my_custom_trigger")
def transform_node(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    # Example: Rewrite 'foo(x)' to 'bar(x, extra=True)'

    # 1. Modify arguments
    new_args = list(node.args)
    new_args.append(cst.Arg(keyword=cst.Name("extra"), value=cst.Name("True")))

    # 2. Modify function name (if needed, though Semantics usually handles this)
    # Note: Generally, let the semantics JSON handle renaming unless logical
    # restructuring is required.

    return node.with_changes(args=new_args)
```

### 2. Register in Semantics

Link your hook to an operation in `semantics/*.json`.

### 3. Usage

When the transpiler encounters the API associated with `my_custom_trigger`, it invokes
your function. The `ctx` object allows you to read configuration or inject code
into the surrounding scope (like adding imports or signature arguments).
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Type, TypeVar, List
import libcst as cst
from pydantic import BaseModel

# We import RuntimeConfig for type hinting
from ml_switcheroo.config import RuntimeConfig

# Validation that SemanticsManager is importable for type hinting
SemanticsManagerType = Any
T = TypeVar("T", bound=BaseModel)

# Callbacks for plugin side-effects
ArgInjectorType = Callable[[str, Optional[str]], None]
PreambleInjectorType = Callable[[str], None]


class HookContext:
  """
  Context object passed to every plugin hook during transcoding.

  This object provides read-only access to the global state and write access
  to specific injection points (signature args, function body preambles).

  Attributes:
      semantics (SemanticsManager): The active knowledge base manager.
      source_fw (str): The import name of the source framework (e.g., 'torch').
      target_fw (str): The import name of the target framework (e.g., 'jax').
      metadata (Dict[str, Any]): A shared dictionary for plugins to store state.
          Use namespaced keys to avoid collisions (e.g. `ctx.metadata['my_plugin']`).
      _runtime_config (RuntimeConfig): The full Pydantic configuration object.
  """

  def __init__(
    self,
    semantics: SemanticsManagerType,
    config: RuntimeConfig,
    arg_injector: Optional[ArgInjectorType] = None,
    preamble_injector: Optional[PreambleInjectorType] = None,
  ):
    """
    Initialize the hook context.

    Args:
        semantics: The loaded SemanticsManager instance.
        config: A populated RuntimeConfig Pydantic model.
        arg_injector: Callback to inject an argument into the enclosing function.
        preamble_injector: Callback to inject code at the start of the enclosing function.
    """
    self.semantics = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector

    # Proxies for convenience in existing plugins
    self.source_fw = config.source_framework
    self.target_fw = config.target_framework
    self.metadata: Dict[str, Any] = {}

  def inject_signature_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """
    Requests the injection of an argument into the current function signature.

    This is useful for threading state arguments (like PRNG keys or variables)
    into functions that didn't previously require them.

    Args:
        name: The argument name (e.g., "rng").
        annotation: Optional type hint (e.g., "jax.random.PRNGKeyArray").

    Example:
        >>> ctx.inject_signature_arg("rng", annotation="jax.Array")
        # Result: def forward(self, x) -> def forward(self, rng: jax.Array, x)
    """
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """
    Requests the injection of a statement at the beginning of the function body.

    Useful for logic that must run before the main operation, such as input
    conversion, RNG splitting, or context management shims.

    Args:
        code_str: The Python statement string. Code must be syntactically valid.

    Example:
        >>> ctx.inject_preamble("rng, key = jax.random.split(rng)")
    """
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: Any = None) -> Any:
    """
    Retrieve a raw value from the unstructured plugin settings dict.

    Args:
        key: Config key (e.g. "epsilon").
        default: Fallback value if key is not present.

    Returns:
        The configuration value or default.
    """
    return self._runtime_config.plugin_settings.get(key, default)

  def config(self, key: str, default: Any = None) -> Any:
    """Legacy alias for raw_config."""
    return self.raw_config(key, default)

  def validate_settings(self, model: Type[T]) -> T:
    """
    Validates the global config against a Plugin-specific Pydantic schema.

    This allows plugins to enforcing strict typing on their required configuration.

    Args:
        model: Pydantic BaseModel class definition defining the expected settings.

    Returns:
        Instance of the model populated with valid data from user config.

    Raises:
        pydantic.ValidationError: If user settings do not match the schema.
    """
    relevant_keys = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant_keys}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> Optional[str]:
    """
    Resolves the target framework's API string for a given standard operation name.

    Args:
        op_name: The standard operation name (e.g., "add", "conv2d").

    Returns:
        The fully qualified API string (e.g. "jax.numpy.add") if found in the
        Semantic Knowledge Base for the current target framework.
    """
    if not self.semantics:
      return None

    known_apis = self.semantics.get_known_apis()
    details = known_apis.get(op_name)
    if not details:
      return None

    variants = details.get("variants", {})
    target_variant = variants.get(self.target_fw)

    if not target_variant:
      return None

    return target_variant.get("api")

  def lookup_signature(self, op_name: str) -> List[str]:
    """
    Retrieves the standard argument list for a given operation.

    Args:
        op_name: The standard operation name (e.g., "add", "sum").

    Returns:
        List of argument names (e.g., ['x', 'axis']). Returns empty list if not found.
        NOTE: This list contains names only, stripping any type information from the spec.
    """
    if not self.semantics:
      return []

    known_apis = self.semantics.get_known_apis()
    details = known_apis.get(op_name)

    if not details:
      return []

    std_args = details.get("std_args", [])

    # Clean up types if they exist: [('x', 'int')] -> ['x']
    cleaned_args = []
    for item in std_args:
      if isinstance(item, (list, tuple)):
        cleaned_args.append(item[0])
      else:
        cleaned_args.append(item)

    return cleaned_args


# Type alias for a Hook Function
HookFunction = Callable[[cst.Call, HookContext], cst.CSTNode]

# Global Registry
_HOOKS: Dict[str, HookFunction] = {}
_PLUGINS_LOADED = False


def register_hook(trigger: str) -> Callable[[HookFunction], HookFunction]:
  """
  Decorator to register a function as a plugin hook.

  Args:
      trigger: The unique identifier string found in `semantics/*.json`
               under the `requires_plugin` field.

  Returns:
      The decorated function (unmodified).
  """

  def decorator(func: HookFunction) -> HookFunction:
    _HOOKS[trigger] = func
    return func

  return decorator


def get_hook(trigger: str) -> Optional[HookFunction]:
  """
  Retrieves a registered hook function by its trigger name.
  Automatically loads plugins if they haven't been loaded yet.

  Args:
      trigger: The plugin identifier string.

  Returns:
      The callable hook function or None if not found.
  """
  if not _PLUGINS_LOADED:
    load_plugins()

  return _HOOKS.get(trigger)


def clear_hooks() -> None:
  """
  Resets the internal hook registry.
  Useful for testing isolation to ensure clean state between tests.
  """
  global _PLUGINS_LOADED
  _HOOKS.clear()
  _PLUGINS_LOADED = False


def load_plugins(plugins_dir: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> int:
  """
  Dynamically imports all modules in the plugins directory and any extra directories.
  This executes the module body, triggering any @register_hook decorators.

  Args:
      plugins_dir: Path object pointing to the specific default plugins folder.
                   If None, defaults to `src/ml_switcheroo/plugins`.
      extra_dirs: List of additional directories to scan for user modules.

  Returns:
      int: Count of modules successfully loaded.
  """
  global _PLUGINS_LOADED

  total_loaded = 0

  # 1. Load Default Plugins (Only once)
  if not _PLUGINS_LOADED:
    target_dir = plugins_dir
    if target_dir is None:
      # Resolve 'src/ml_switcheroo/plugins' based on this file's location
      current_dir = Path(__file__).resolve().parent
      target_dir = current_dir.parent / "plugins"

    # Fallback to package lookup if calculated path fails
    if not target_dir.exists():
      try:
        import ml_switcheroo.plugins

        if ml_switcheroo.plugins.__file__:
          target_dir = Path(ml_switcheroo.plugins.__file__).parent
      except (ImportError, AttributeError):
        pass

    if target_dir and target_dir.exists():
      total_loaded += _import_from_dir(target_dir, base_package="ml_switcheroo.plugins")
      _PLUGINS_LOADED = True

  # 2. Load External Plugins (Always scan if provided, allowing overrides)
  if extra_dirs:
    for ex_dir in extra_dirs:
      if ex_dir.exists() and ex_dir.is_dir():
        total_loaded += _import_from_dir(ex_dir, base_package=None)

  return total_loaded


def _import_from_dir(directory: Path, base_package: Optional[str] = None) -> int:
  """
  Helper to iterate and import python files from a directory.

  Args:
      directory: The folder to scan.
      base_package: Optional python package import prefix.

  Returns:
      Count of imported modules.
  """
  count = 0
  for item in directory.glob("*.py"):
    if item.name == "__init__.py":
      continue

    module_name = item.stem

    # Try standard import if package structure exists
    if base_package:
      try:
        if base_package in sys.modules or (
          sys.modules.get("ml_switcheroo") and hasattr(sys.modules["ml_switcheroo"], "plugins")
        ):
          importlib.import_module(f"{base_package}.{module_name}")
          count += 1
          continue
      except Exception:
        pass

    # Robust File Import
    try:
      # Create a unique name to avoid collision if two user dirs have 'my_plugin.py'
      unique_name = f"switcheroo_plugin_{module_name}_{item.stat().st_ino}"
      spec = importlib.util.spec_from_file_location(unique_name, item)
      if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
        count += 1
    except Exception as e:
      print(f"Failed to load plugin {item.name}: {e}")

  return count
