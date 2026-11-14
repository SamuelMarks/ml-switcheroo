"""Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement,
supporting both standard Python frameworks and custom markup/DSL ingestion hooks
via `CustomParserProtocol` and Pipeline Routing.
"""

import importlib
import importlib.util
import json
import logging
import os
import typing
from enum import Enum
from functools import lru_cache
from pathlib import Path

import libcst as cst
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier, StandardMap
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ml_switcheroo.core.dsl import OperationDef as OperationDef
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.exceptions import ConfigurationError
from ml_switcheroo.semantics.schema import PluginTraits, StructuralTraits

SNAPSHOT_DIR: Path | None = None


def resolve_snapshot_dir(snapshot_dir: Path | None = None) -> Path | None:
  """Resolve the snapshot directory from explicit argument, override, env var, or installed package.

  Args:
      snapshot_dir: Optional explicit snapshot directory path.

  Returns:
      Path | None: Resolved directory path if configured or installed, otherwise ``None``.
  """
  if snapshot_dir is not None:
    return snapshot_dir
  if SNAPSHOT_DIR is not None:
    return SNAPSHOT_DIR
  env_dir = os.environ.get("ML_SWITCHEROO_SNAPSHOT_DIR", "").strip()
  if env_dir:
    return Path(env_dir)
  try:
    spec = importlib.util.find_spec("ml_ecosystem_snapshots")
  except (ImportError, ValueError, AttributeError):
    spec = None
  if spec is not None and spec.origin:
    return Path(spec.origin).resolve().parent
  return None


@typing.runtime_checkable
class CustomParserProtocol(typing.Protocol):
  """Protocol for custom DSL parsers that emit a Python LibCST Module."""

  def parse(self) -> cst.Module:
    """Parse raw DSL source into a Python LibCST Module.

    Returns:
        cst.Module: The parsed Python CST module.
    """
    ...


class InitMode(str, Enum):
  """Initialize mode for adapters."""

  LIVE = "live"
  GHOST = "ghost"


class ImportConfig(BaseModel):
  """Configuration for an exposed namespace."""

  tier: SemanticTier = Field(description="The semantic category of this namespace.")
  recommended_alias: str | None = Field(default=None, description="Preferred alias (e.g., 'nn').")


class FrameworkAdapter(typing.Protocol):
  """Protocol definition for a Framework Adapter.

  Defines the comprehensive set of properties, configuration options,
  and code generation methods required to integrate a machine learning framework
  into the translation and analysis pipelines.
  """

  def __init__(self) -> None:
    """Initialize the framework adapter instance."""
    ...

  def convert(self, data: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
    """Convert framework-specific or generic data structures.

    Args:
        data: The input framework-specific or raw data structure to be converted.

    Returns:
        The converted representation compatible with the pipeline.
    """
    ...

  @property
  def test_config(self) -> dict[str, str]:
    """Retrieve testing-specific configurations for the adapter.

    Returns:
        A dictionary of test configurations, such as framework-specific
        test flags, tolerances, or environment overrides.
    """
    ...

  @property
  def harness_imports(self) -> list[str]:
    """Get the necessary module import statements for generating a test harness.

    Returns:
        A list of import strings (e.g., ["import torch", "import torch.nn as nn"]).
    """
    ...

  def get_harness_init_code(self) -> str:
    """Generate framework-specific initialization code for the test harness.

    Returns:
        A string containing Python code to initialize the framework environment
        (e.g., setting global states or precision).
    """
    ...

  def get_to_numpy_code(self) -> str:
    """Retrieve the Python code snippet used to convert a framework tensor to a NumPy array.

    Returns:
        A code string defining helper logic to convert variables to NumPy arrays.
    """
    ...

  @property
  def display_name(self) -> str:
    """Return the user-facing display name of the framework.

    Returns:
        The display name of the framework (e.g., "PyTorch", "JAX").
    """
    ...

  @property
  def ui_priority(self) -> int:
    """Return the sorting or rendering priority of this framework in UI views.

    Returns:
        An integer indicating priority. Higher values represent higher priority.
    """
    ...

  @property
  def supported_tiers(self) -> list[SemanticTier]:
    """Get the semantic tiers supported by this framework adapter.

    Returns:
        A list of supported SemanticTier values (e.g., TENSOR, NEURAL_NETWORK).
    """
    ...

  @property
  def import_alias(self) -> tuple[str, str]:
    """Return the standard import name and recommended package alias for this framework.

    Returns:
        A tuple of (module_name, recommended_alias), e.g., ("torch", "torch").
    """
    ...

  @property
  def inherits_from(self) -> str | None:
    """Identify if this framework adapter inherits behavior from another adapter.

    Returns:
        The name of the parent framework if one exists, otherwise None.
    """
    ...

  @property
  def structural_traits(self) -> StructuralTraits:
    """Retrieve structural traits and preferences of the framework.

    Returns:
        The structural traits configuration of the framework, dictating
        syntactic constraints like keyword-only arguments or native array layouts.
    """
    ...

  @property
  def plugin_traits(self) -> PluginTraits:
    """Retrieve plugin-specific traits and capabilities supported by the framework.

    Returns:
        The plugin traits configuration highlighting enabled transform plugins.
    """
    ...

  @property
  def rng_seed_methods(self) -> list[str]:
    """List the framework APIs or methods used to set random number generator seeds.

    Returns:
        A list of seeding method names or statements.
    """
    ...

  @property
  def declared_magic_args(self) -> list[str]:
    """Return special or magic argument names that require translation overrides.

    Returns:
        A list of argument names that require custom mapping logic (e.g., 'device', 'dtype').
    """
    ...

  def get_device_syntax(self, device_type: str, device_index: str | None = None) -> str:
    """Generate framework-specific syntax for device placement.

    Args:
        device_type: The device type target (e.g., "cuda", "cpu", "mps").
        device_index: Optional string representing the specific device ID or index.

    Returns:
        A string representing the device instantiation/placement syntax.
    """
    ...

  def get_device_check_syntax(self) -> str:
    """Generate framework-specific code for querying/verifying available devices.

    Returns:
        A Python code snippet that checks and returns device status or availability.
    """
    ...

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Generate the syntax for splitting random number generator states/keys.

    Args:
        rng_var: Variable name of the source RNG state or key.
        key_var: Variable name where the split keys will be assigned.

    Returns:
        A code snippet representing the RNG split operation.
    """
    ...

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: str | None = None) -> str:
    """Generate framework-specific syntax for serializing or deserializing models/weights.

    Args:
        op: Serialization operation type, typically "save" or "load".
        file_arg: File path variable or string representation.
        object_arg: Optional variable name of the object to be saved/serialized.

    Returns:
        A Python code snippet executing the serialization action.
    """
    ...

  def get_serialization_imports(self) -> list[str]:
    """Get the necessary module imports for model/weight serialization operations.

    Returns:
        A list of required import statement strings.
    """
    ...

  def get_weight_conversion_imports(self) -> list[str]:
    """Get the necessary module imports for weight format conversion.

    Returns:
        A list of required import statement strings.
    """
    ...

  def get_weight_load_code(self, path_var: str) -> str:
    """Generate framework-specific code to load weights from a file path.

    Args:
        path_var: Variable containing the source weight file path.

    Returns:
        A code snippet to load the weights into the framework's state dictionary/model.
    """
    ...

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generate an expression converting a framework tensor/array to a NumPy array.

    Args:
        tensor_var: Variable name of the framework tensor.

    Returns:
        The expression code string (e.g., "tensor_var.detach().cpu().numpy()").
    """
    ...

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generate framework-specific code to save weights/states to a file path.

    Args:
        state_var: Variable containing model weights or state dictionary.
        path_var: Variable containing target destination file path.

    Returns:
        A code snippet to save the weights/state.
    """
    ...

  def get_doc_url(self, api_name: str) -> str | None:
    """Retrieve the official documentation URL for a given framework API.

    Args:
        api_name: The name or path of the framework API.

    Returns:
        The documentation URL string if found, otherwise None.
    """
    ...

  def get_tiered_examples(self) -> dict[str, str]:
    """Provide standard usage examples for different semantic tiers.

    Returns:
        A dictionary mapping semantic tier or category names to example code snippets.
    """
    ...

  @property
  def definitions(self) -> dict[str, StandardMap]:
    """Return the semantic mapping definitions of framework APIs to the IR standard map.

    Returns:
        A dictionary mapping framework API names to their StandardMap definitions.
    """
    ...

  @property
  def specifications(self) -> dict[str, OperationDef]:
    """Retrieve the full operation signatures and specifications for framework APIs.

    Returns:
        A dictionary mapping operation names to their OperationDef specifications.
    """
    ...

  @property
  def import_namespaces(self) -> dict[str, dict[str, str] | ImportConfig]:
    """Return the configured namespaces and import rules exposed by this framework.

    Returns:
        A dictionary mapping namespace identifiers to import configurations or maps.
    """
    ...

  def apply_wiring(self, snapshot: dict[str, dict[str, JsonValue]]) -> None:
    """Apply dynamic bindings or overrides onto the adapter using snapshot data.

    Args:
        snapshot: A dictionary containing API binding snapshot data.
    """
    ...


def load_snapshot_for_adapter(fw_key: str, snapshot_dir: Path | None = None) -> dict[str, JsonValue]:
  """Load the most recent snapshot JSON data for ``fw_key`` from the resolved snapshot directory.

  Args:
      fw_key: The identifier key representing the framework adapter.
      snapshot_dir: Optional explicit directory containing ``<fw_key>_v*.json`` files.

  Returns:
      The parsed JSON snapshot dictionary, or an empty dictionary if no snapshots
      are found or if loading fails.
  """
  resolved_dir = resolve_snapshot_dir(snapshot_dir)
  if resolved_dir is None or not resolved_dir.exists():
    return {}
  candidates = sorted(list(resolved_dir.glob(f"{fw_key}_v*.json")))
  if not candidates:
    return {}
  target = candidates[-1]
  try:
    with open(target, "r", encoding="utf-8") as f:
      data = json.load(f)
      return data if isinstance(data, dict) else {}
  except (json.JSONDecodeError, OSError) as e:
    logging.error(f"Failed to load snapshot {target}: {e}")
    return {}


class FrameworkSnapshot(BaseModel):
  """Typed Ghost Protocol snapshot container wrapping validated ``GhostRef`` entries.

  Attributes:
      version: Framework version string recorded in the snapshot envelope, if any.
      categories: Mapping from semantic tier key to validated ``GhostRef`` definitions.
  """

  model_config = ConfigDict(extra="ignore")

  version: str = ""
  categories: dict[str, list[GhostRef]] = Field(default_factory=dict)

  def is_empty(self) -> bool:
    """Check whether the snapshot holds neither categorized entries nor version metadata.

    Returns:
        True when both ``categories`` and ``version`` are empty.
    """
    return not self.categories and not self.version

  @classmethod
  def from_raw(
    cls,
    raw: "FrameworkSnapshot | typing.Mapping[str, JsonValue] | None",
  ) -> "FrameworkSnapshot":
    """Validate an arbitrary snapshot mapping or ``FrameworkSnapshot`` into a typed model.

    Args:
        raw: Optional ``FrameworkSnapshot`` instance or raw JSON mapping.

    Returns:
        Validated ``FrameworkSnapshot`` wrapping ``GhostRef`` entries.
    """
    if isinstance(raw, cls):
      return raw
    if not isinstance(raw, typing.Mapping) or not raw:
      return cls()
    raw_version = raw.get("version")
    version_str = (
      str(raw_version) if isinstance(raw_version, (str, int, float)) and not isinstance(raw_version, bool) else ""
    )
    parsed_cats: dict[str, list[GhostRef]] = {}
    raw_cats = raw.get("categories")
    if isinstance(raw_cats, typing.Mapping):
      for cat_key, raw_list in raw_cats.items():
        if isinstance(raw_list, list):
          parsed_cats[str(cat_key)] = [GhostRef.model_validate(item) for item in raw_list if isinstance(item, dict)]
    return cls(version=version_str, categories=parsed_cats)

  def get_category(self, category: SemanticTier) -> list[GhostRef]:
    """Retrieve validated ``GhostRef`` entries for ``category``.

    Args:
        category: Semantic tier category to extract.

    Returns:
        List of ``GhostRef`` entries for ``category``.
    """
    return list(self.categories.get(category.value, []))


class SnapshotBackedAdapter:
  """Composition helper managing live/ghost initialization mode and typed snapshot lookup.

  Attributes:
      framework_key: Canonical framework key used when loading fallback snapshots.
      mode: Active initialization mode (``InitMode.LIVE`` or ``InitMode.GHOST``).
      snapshot: Validated ``FrameworkSnapshot`` model loaded in ghost mode.
      snapshot_data: Alias to ``snapshot`` holding the typed ``FrameworkSnapshot`` model.
  """

  def __init__(
    self,
    framework_key: str,
    *,
    is_live_available: bool = True,
    loader: typing.Callable[[str], FrameworkSnapshot | dict[str, JsonValue] | None] | None = None,
    missing_message: str | None = None,
  ) -> None:
    """Initialize snapshot-backed state for a framework adapter.

    Args:
        framework_key: Canonical framework identifier (e.g., ``"torch"``, ``"jax"``).
        is_live_available: Whether the live Python framework package is importable.
        loader: Optional snapshot loader override (defaults to ``load_snapshot_for_adapter``).
        missing_message: Optional debug log message emitted when ghost snapshot data is absent.
    """
    self.framework_key: str = framework_key
    if is_live_available:
      self.mode: InitMode = InitMode.LIVE
      self.snapshot: FrameworkSnapshot = FrameworkSnapshot()
    else:
      self.mode = InitMode.GHOST
      active_loader = loader if loader is not None else load_snapshot_for_adapter
      loaded = active_loader(framework_key)
      self.snapshot = FrameworkSnapshot.from_raw(loaded)
      if (
        loaded is None or loaded == {} or (isinstance(loaded, FrameworkSnapshot) and loaded.is_empty())
      ) and missing_message:
        logging.debug(missing_message)
    self.snapshot_data: FrameworkSnapshot = self.snapshot

  @staticmethod
  def collect_ghost_from(
    snapshot_data: FrameworkSnapshot | typing.Mapping[str, JsonValue] | None,
    category: SemanticTier,
  ) -> list[GhostRef]:
    """Hydrate ``GhostRef`` entries for ``category`` from a ``FrameworkSnapshot`` or mapping.

    Args:
        snapshot_data: Typed ``FrameworkSnapshot``, raw snapshot mapping, or ``None``.
        category: Semantic tier category to extract.

    Returns:
        Validated list of ``GhostRef`` models for ``category``.
    """
    if snapshot_data is None:
      return []
    return FrameworkSnapshot.from_raw(snapshot_data).get_category(category)

  def collect_ghost(self, category: SemanticTier) -> list[GhostRef]:
    """Hydrate ``GhostRef`` entries for ``category`` from ``self.snapshot``.

    Args:
        category: Semantic tier category to extract.

    Returns:
        Validated list of ``GhostRef`` models for ``category``.
    """
    return self.snapshot.get_category(category)


AdapterT = typing.TypeVar("AdapterT")


def _is_adapter_class(
  cls: AdapterT | type[FrameworkAdapter],
) -> typing.TypeGuard[type[FrameworkAdapter]]:
  """Verify whether a decorated object is a valid framework adapter class.

  Args:
      cls: Candidate class being registered in the adapter registry.

  Returns:
      True if ``cls`` is a class type, narrowing it to ``type[FrameworkAdapter]``.
  """
  return isinstance(cls, type)


class AdapterRegistry:
  """Explicit registry encapsulating ``FrameworkAdapter`` classes and cached adapter instances.

  Attributes:
      adapters: Mapping from canonical framework identifiers to ``FrameworkAdapter`` classes.
  """

  def __init__(self) -> None:
    """Initialize an empty framework adapter registry."""
    self.adapters: dict[str, type[FrameworkAdapter]] = {}
    self._instances: dict[str, FrameworkAdapter] = {}

  def register_class(self, name: str, cls: type[FrameworkAdapter]) -> None:
    """Register an adapter class under ``name`` and invalidate any cached instance.

    Args:
        name: Canonical framework identifier.
        cls: Concrete ``FrameworkAdapter`` implementation class.
    """
    self.adapters[name] = cls
    self._instances.pop(name, None)

  def register(self, name: str) -> typing.Callable[[AdapterT], AdapterT]:
    """Create a class decorator registering a ``FrameworkAdapter`` implementation under ``name``.

    Args:
        name: Canonical framework identifier.

    Returns:
        Decorator returning the registered class unchanged.
    """

    def wrapper(cls: AdapterT) -> AdapterT:
      """Validate and register ``cls`` in this ``AdapterRegistry``.

      Args:
          cls: Concrete ``FrameworkAdapter`` implementation class.

      Returns:
          Unchanged ``cls``.

      Raises:
          ConfigurationError: If ``cls`` is not a class object.
      """
      reg_cls: AdapterT | type[FrameworkAdapter] = cls
      if not _is_adapter_class(reg_cls):
        raise ConfigurationError(f"Adapter registered for '{name}' must be a class.")
      self.register_class(name, reg_cls)
      return cls

    return wrapper

  def available_frameworks(self) -> list[str]:
    """Return all registered framework identifiers.

    Returns:
        List of registered framework names.
    """
    return list(self.adapters.keys())

  def get_adapter_class(self, name: str) -> type[FrameworkAdapter] | None:
    """Look up the registered adapter class for ``name`` (or ``{name}_nnx`` fallback).

    Args:
        name: Registered framework name.

    Returns:
        Registered ``FrameworkAdapter`` class if found, else ``None``.
    """
    return self.adapters.get(name) or self.adapters.get(f"{name}_nnx")

  def get_adapter(self, name: str) -> FrameworkAdapter | None:
    """Return a cached ``FrameworkAdapter`` instance for ``name``, instantiating lazily on first request.

    Args:
        name: Registered framework name.

    Returns:
        Cached ``FrameworkAdapter`` instance if registered, otherwise ``None``.
    """
    if name in self._instances:
      return self._instances[name]
    cls = self.get_adapter_class(name)
    if cls is not None:
      instance: FrameworkAdapter = cls()
      self._instances[name] = instance
      return instance
    return None

  def clear_cache(self) -> None:
    """Clear all cached ``FrameworkAdapter`` instances."""
    self._instances.clear()

  def snapshot(self) -> dict[str, type[FrameworkAdapter]]:
    """Return a shallow copy of the registered adapter classes mapping.

    Returns:
        Copy of ``self.adapters``.
    """
    return self.adapters.copy()

  def restore(self, state: typing.Mapping[str, type[FrameworkAdapter]]) -> None:
    """Restore registered adapter classes from ``state`` and clear cached instances.

    Args:
        state: Saved mapping of framework names to adapter classes.
    """
    self.adapters.clear()
    self.adapters.update(state)
    self._instances.clear()


@lru_cache(maxsize=1)
def get_default_adapter_registry() -> AdapterRegistry:
  """Return the lazily instantiated default ``AdapterRegistry``.

  Returns:
      Default ``AdapterRegistry`` instance.
  """
  return AdapterRegistry()


def register_framework(name: str) -> typing.Callable[[AdapterT], AdapterT]:
  """Get a decorator to register concrete FrameworkAdapter classes under a specific name.

  Args:
      name: The global registration key/name for the framework.

  Returns:
      A decorator function that maps the decorated class to the given name
      in the default ``AdapterRegistry`` and returns the class itself.
  """
  return get_default_adapter_registry().register(name)


def available_frameworks() -> list[str]:
  """Retrieve the names of all currently registered framework adapters.

  Returns:
      A list of registered framework adapter keys/names.
  """
  return get_default_adapter_registry().available_frameworks()


def get_adapter_class(name: str) -> type[FrameworkAdapter] | None:
  """Return the registered ``FrameworkAdapter`` class for ``name`` without instantiating it.

  Args:
      name: Registered name of the framework adapter.

  Returns:
      Registered ``FrameworkAdapter`` class if found, else ``None``.
  """
  return get_default_adapter_registry().get_adapter_class(name)


def get_adapter(name: str) -> FrameworkAdapter | None:
  """Instantiate and return the registered FrameworkAdapter subclass corresponding to the given name.

  Args:
      name: The registered name of the framework adapter.

  Returns:
      An instance of FrameworkAdapter if registered, otherwise None.
  """
  return get_default_adapter_registry().get_adapter(name)


__all__ = [
  "AdapterRegistry",
  "CustomParserProtocol",
  "FrameworkAdapter",
  "FrameworkSnapshot",
  "GhostRef",
  "ImportConfig",
  "InitMode",
  "PluginTraits",
  "SemanticTier",
  "SnapshotBackedAdapter",
  "StandardMap",
  "StructuralTraits",
  "available_frameworks",
  "get_adapter",
  "get_adapter_class",
  "get_default_adapter_registry",
  "load_snapshot_for_adapter",
  "register_framework",
]
