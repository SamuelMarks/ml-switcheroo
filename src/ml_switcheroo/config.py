"""Runtime Configuration Store.

Supports Dynamic Defaults, TOML loading, and Framework Flavours.

This module resolves default Source and Target frameworks by querying the
``ui_priority`` of registered adapters, ensuring the logic is completely agnostic
to the specific libraries installed.
"""

import contextlib
import sys
import typing
from enum import Enum
from pathlib import Path as Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from ml_switcheroo.exceptions import ConfigurationError

IntermediateFormat: typing.TypeAlias = typing.Literal[
  "ir",
  "ml_switcheroo_ir",
  "mlir",
  "stablehlo",
  "python",
]

VALID_INTERMEDIATE_FORMATS: frozenset[str] = frozenset(
  {
    "ir",
    "ml_switcheroo_ir",
    "mlir",
    "stablehlo",
    "python",
  }
)


def validate_intermediate_format(value: str | None) -> IntermediateFormat | None:
  """Validate and normalize an optional intermediate representation identifier.

  Args:
      value: Candidate intermediate dialect string or ``None``.

  Returns:
      Validated ``IntermediateFormat`` literal or ``None``.

  Raises:
      ConfigurationError: If ``value`` is not a supported intermediate format.
  """
  if value is None:
    return None
  cleaned = value.strip().lower()
  if cleaned == "ir":
    return "ir"
  if cleaned == "ml_switcheroo_ir":
    return "ml_switcheroo_ir"
  if cleaned == "mlir":
    return "mlir"
  if cleaned == "stablehlo":
    return "stablehlo"
  if cleaned == "python":
    return "python"
  supported = ", ".join(sorted(VALID_INTERMEDIATE_FORMATS))
  raise ConfigurationError(f"Unsupported intermediate format '{value}'. Supported formats: {supported}.")


class ConversionMode(str, Enum):
  """Transpilation mode controlling trivia preservation vs multi-pass graph optimization."""

  PRESERVE_TRIVIA = "preserve_trivia"
  OPTIMIZED = "optimized"


class OptimizationLevel(str, Enum):
  """Compiler pass optimization level when running in `ConversionMode.OPTIMIZED`."""

  O0 = "O0"
  O1 = "O1"
  O2 = "O2"
  O3 = "O3"


class ExecutionBackend(str, Enum):
  """Execution backend for forward and backward passes via `ml-switcheroo-compiler`."""

  # Web-Native / Edge Backends (Browser & Portable)
  WEBGPU = "webgpu"
  WASM_SIMD = "wasm_simd"
  WEBGL = "webgl"
  PYODIDE = "pyodide"
  AUTO = "auto"
  ONNX = "onnx"
  STABLEHLO = "stablehlo"

  # Hardware Accelerator Backends
  CUDA = "cuda"
  ROCM = "rocm"
  METAL = "metal"
  LLVM_CPP = "llvm_cpp"

  # Array & Framework Compiler Backends
  JAX = "jax"
  PYTORCH = "pytorch"
  TORCH = "torch"
  MLX = "mlx"
  TENSORFLOW = "tensorflow"
  KERAS = "keras"
  NUMPY = "numpy"
  CUPY = "cupy"
  DASK = "dask"
  NUMBA = "numba"
  DPNP = "dpnp"
  SPARSE = "sparse"
  SPARSE_COO = "sparse_coo"
  AWKWARD = "awkward"
  PYARROW_COMPUTE = "pyarrow_compute"
  PURE_PYTHON = "pure_python"


class LossReduction(str, Enum):
  """Scalar loss reduction applied to forward output for backward pass."""

  SUM = "sum"
  MEAN = "mean"
  L2 = "l2"


def parse_input_shapes_string(raw: str) -> list[tuple[int, ...]] | None:
  """Parse UI input shape strings (e.g., '2, 10' or '(2, 3, 8, 8); (2, 10)') into tuples.

  Args:
      raw: Raw shape specification string from UI or CLI.

  Returns:
      List of integer shape tuples, or None if no valid dimensions are present.
  """
  cleaned = raw.strip()
  if not cleaned or cleaned.lower() == "auto":
    return None

  shapes: list[tuple[int, ...]] = []
  for group in cleaned.split(";"):
    stripped_group = group.strip().strip("()[]{}")
    if not stripped_group:
      continue
    dims: list[int] = []
    for token in stripped_group.split(","):
      tok = token.strip()
      if not tok:
        continue
      try:
        val = int(tok)
      except ValueError:
        return None
      if val <= 0:
        return None
      dims.append(val)
    if dims:
      shapes.append(tuple(dims))

  return shapes if shapes else None


class TomlLoaderProtocol(typing.Protocol):
  """Protocol for TOML parser modules (tomllib / tomli)."""

  def load(self, fp: typing.BinaryIO) -> dict[str, JsonValue]:
    """Load TOML data from a binary file object.

    Args:
        fp: Binary file stream.

    Returns:
        Parsed TOML dictionary.
    """
    ...


# Optional TOML support
def _import_tomllib() -> TomlLoaderProtocol | None:
  """Import tomllib or tomli depending on python version.

  Returns:
      Optional[TomlLoaderProtocol]: The imported tomllib/tomli module, or None if neither is available.
  """
  if sys.version_info >= (3, 11):
    import tomllib

    loader: TomlLoaderProtocol | None = tomllib
  else:
    try:
      import tomli

      loader = tomli
    except ImportError:
      loader = None
  return loader


tomllib = _import_tomllib()

T = typing.TypeVar("T", bound=BaseModel)


def get_framework_priority_order() -> list[str]:
  """Return a list of framework keys sorted by UI Priority.

  The sort order is determined by the ``ui_priority`` attribute of the
  registered FrameworkAdapter. Lower numbers appear first (Source default).

  If no priority is defined, defaults to 999.

  Returns:
      List[str]: Sorted list of framework identifiers (e.g. ['torch', 'jax']).

  """
  # Defer import to prevent circular dependency during initialization
  # config -> frameworks/__init__ -> nvidia_sass -> hooks -> config
  from ml_switcheroo.frameworks.base import available_frameworks, get_adapter

  frameworks = available_frameworks()

  def sort_key(name: str) -> tuple[int, str]:
    """Support to generate a sorting key for a framework name.

    Args:
        name (str): The framework name/key.

    Returns:
        Tuple[int, str]: A tuple of (priority, name) for sorting.
    """
    adapter = get_adapter(name)
    priority = 999
    is_child = False

    if adapter:
      # Check hierarchy to push flavours/children to the end
      try:
        if adapter.inherits_from:
          is_child = True
      except AttributeError:
        pass

      # Extract priority
      with contextlib.suppress(AttributeError, ValueError, TypeError):
        priority = int(adapter.ui_priority)

    # Move children (flavours) to the very end for UI clarity
    if is_child:
      priority = 9999

    # Tuple sort: Priority first, then Alphabetical tie-break
    return (priority, name)

  return sorted(frameworks, key=sort_key)


def _resolve_default_source() -> str:
  """Resolve the default source framework.

  Returns:
      str: The highest priority framework key (Index 0 in sorted list),
      or "source_placeholder" if none are registered.

  """
  fws = get_framework_priority_order()
  return fws[0] if fws else "source_placeholder"


def default_source_framework() -> str:
  """Resolve the highest-priority registered source framework key.

  Returns:
      str: Highest-priority registered framework key.
  """
  return _resolve_default_source()


def _resolve_default_target() -> str:
  """Resolve the default target framework.

  Returns:
      str: The second highest priority framework key (Index 1),
      or the first (Index 0) if only one exists, or "target_placeholder".

  """
  fws = get_framework_priority_order()
  if len(fws) >= 2:
    return fws[1]
  return fws[0] if fws else "target_placeholder"


class RuntimeConfig(BaseModel):
  """Global configuration container for the translation engine.

  Attributes:
      source_framework (str): The primary source framework key.
      target_framework (str): The primary target framework key.
      source_flavour (Optional[str]): Detailed source sub-framework.
      target_flavour (Optional[str]): Detailed target sub-framework.
      strict_mode (bool): If True, fail on unknown APIs.
      intermediate (Optional[str]): Intermediate representation layer (e.g. 'ir', 'mlir', 'tikz') for round-trip verification.
      enable_graph_optimization (bool): If True, performs graph-level optimization and fusion.
      enable_import_fixer (bool): If True, performs import resolution, pruning, and injection.
      enable_sharding (bool): If True, enables distributed sharding logic and related fusions.
      plugin_settings (Dict[str, ScalarValue]): Configuration passed to plugins.
      plugin_paths (List[Path]): External directories to scan for plugins.
      validation_report (Optional[Path]): Path to a verification lockfile.
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
  source_flavour: str | None = Field(None, description="Detailed source sub-framework.")
  target_flavour: str | None = Field(None, description="Detailed target sub-framework.")

  strict_mode: bool = Field(False, description="If True, fail on unknown APIs.")

  # Structural Verification
  intermediate: IntermediateFormat | None = Field(
    None,
    description="Intermediate representation layer ('ir', 'ml_switcheroo_ir', 'mlir', 'stablehlo', 'python') for round-trip execution.",
  )

  # Pipeline Control Flags
  mode: ConversionMode = Field(
    default=ConversionMode.PRESERVE_TRIVIA,
    description="Conversion mode: 'preserve_trivia' (1:1 whitespace/comments) or 'optimized' (multi-pass compiler optimizations).",
  )
  opt_level: OptimizationLevel = Field(
    default=OptimizationLevel.O2,
    description="Compiler pass optimization level (O0, O1, O2, O3) when running in 'optimized' mode.",
  )
  enable_graph_optimization: bool = Field(False, description="If True, performs graph-level optimization and fusion.")
  enable_import_fixer: bool = Field(True, description="If True, performs import resolution, pruning, and injection.")
  enable_sharding: bool = Field(False, description="If True, enables distributed sharding logic and related fusions.")

  plugin_settings: dict[str, JsonValue] = Field(default_factory=dict, description="Configuration passed to plugins.")
  plugin_paths: list[Path] = Field(default_factory=list, description="External directories to scan for plugins.")
  validation_report: Path | None = Field(None, description="Path to a verification lockfile.")
  snapshot_dir: Path | None = Field(None, description="Optional directory containing framework API snapshots.")
  workspace_path: Path | None = Field(None, description="Optional workspace directory for sibling development packages.")

  # On-Device Execution & Autodiff Configuration
  enable_on_device_execution: bool = Field(
    default=False,
    description="Execute forward and backward passes on-device via zero-* and ml-switcheroo-compiler.",
  )
  execution_backend: ExecutionBackend = Field(
    default=ExecutionBackend.WEBGPU,
    description="Execution backend for forward/backward passes (defaults to web-native 'webgpu').",
  )
  input_shapes: list[tuple[int, ...]] | None = Field(
    default=None,
    description="Explicit input tensor shapes overriding AST auto-inference.",
  )
  input_dtype: str = Field(
    default="float32",
    description="Data type for generated on-device input tensors.",
  )
  random_seed: int = Field(
    default=42,
    description="Deterministic seed for on-device input and parameter generation.",
  )
  loss_reduction: LossReduction = Field(
    default=LossReduction.SUM,
    description="Scalar loss reduction applied to forward output for backward pass.",
  )
  inject_runtime_annotations: bool = Field(
    default=True,
    description="Inject inline forward/backward shape and gradient comments into target code.",
  )
  inject_shape_assertions: bool = Field(
    default=False,
    description="Inject executable shape assertions into target code.",
  )
  specialize_dynamic_branches: bool = Field(
    default=False,
    description="Replace tensor-conditioned if/else statements with the branch taken during on-device execution, preserving untaken branch as comments.",
  )
  emit_backward_code: bool = Field(
    default=False,
    description="Synthesize explicit target-framework backward/VJP function from the autodiff LogicalGraph.",
  )
  observed_runtime_shapes: dict[str, list[int]] | None = Field(
    default=None,
    description="Pre-captured hardware shapes from WebGPU/WASM browser runners.",
  )

  @model_validator(mode="after")
  def _sync_mode_and_graph_optimization(self) -> "RuntimeConfig":
    """Synchronize `mode` and `enable_graph_optimization` fields.

    Returns:
        RuntimeConfig: Synchronized configuration instance.
    """
    if self.mode == ConversionMode.OPTIMIZED and not self.enable_graph_optimization:
      self.enable_graph_optimization = True
    elif self.enable_graph_optimization and self.mode == ConversionMode.PRESERVE_TRIVIA:
      self.mode = ConversionMode.OPTIMIZED
    return self

  @field_validator("intermediate", mode="before")
  @classmethod
  def validate_intermediate(cls, v: str | None) -> IntermediateFormat | None:
    """Validate that ``intermediate`` is a supported intermediate representation.

    Args:
        v: Raw intermediate format string or ``None``.

    Returns:
        Validated ``IntermediateFormat`` literal or ``None``.

    Raises:
        ConfigurationError: If ``v`` is not a supported intermediate representation.
    """
    return validate_intermediate_format(v)

  @field_validator("source_framework", "target_framework")
  @classmethod
  def validate_framework(cls, v: str) -> str:
    """Ensure the framework is registered in the system.

    Args:
        v (str): The framework key to validate.

    Returns:
        str: The normalized framework key.

    Raises:
        ValueError: If the framework is not registered and not a placeholder.

    """
    # Defer import to prevent circular dependency
    from ml_switcheroo.frameworks.base import available_frameworks

    v_clean = v.lower().strip()
    known = available_frameworks()
    # Allow unregistered checks if registry is empty (bootstrap/test mode)
    # or if using the placeholders generated when empty.
    if known and v_clean not in known and v_clean not in ["source_placeholder", "target_placeholder"]:
      raise ValueError(f"Unknown framework: '{v_clean}'. Supported frameworks: {known}")
    return v_clean

  @property
  def effective_source(self) -> str:
    """Return the resolved source framework key.

    Prioritizes ``source_flavour`` if present, otherwise returns ``source_framework``.

    Returns:
        str: The active source framework key.

    """
    return self.source_flavour if self.source_flavour else self.source_framework

  @property
  def effective_target(self) -> str:
    """Return the resolved target framework key.

    Prioritizes ``target_flavour`` if present, otherwise returns ``target_framework``.

    Returns:
        str: The active target framework key.

    """
    return self.target_flavour if self.target_flavour else self.target_framework

  def parse_plugin_settings(self, schema: type[T]) -> T:
    """Validate plugin settings against a Pydantic model.

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
      raise ValueError(f"Plugin configuration validation failed: {e}") from e

  @staticmethod
  def _resolve_frameworks_and_flavours(
    toml_config: typing.Mapping[str, JsonValue],
    source: str | None,
    target: str | None,
    source_flavour: str | None,
    target_flavour: str | None,
  ) -> tuple[str, str, str | None, str | None]:
    """Merge CLI overrides, TOML entries, and dynamic defaults for source/target frameworks and flavours.

    Args:
        toml_config: Parsed ``[tool.ml_switcheroo]`` mapping.
        source: CLI source framework override.
        target: CLI target framework override.
        source_flavour: CLI source flavour override.
        target_flavour: CLI target flavour override.

    Returns:
        Tuple of ``(source_framework, target_framework, source_flavour, target_flavour)``.
    """
    raw_src = toml_config.get("source_framework")
    final_source = source if source else (str(raw_src) if isinstance(raw_src, str) else _resolve_default_source())
    raw_tgt = toml_config.get("target_framework")
    final_target = target if target else (str(raw_tgt) if isinstance(raw_tgt, str) else _resolve_default_target())

    raw_src_flav = toml_config.get("source_flavour")
    final_src_flavour = source_flavour or (str(raw_src_flav) if isinstance(raw_src_flav, str) else None)
    raw_tgt_flav = toml_config.get("target_flavour")
    final_tgt_flavour = target_flavour or (str(raw_tgt_flav) if isinstance(raw_tgt_flav, str) else None)
    return final_source, final_target, final_src_flavour, final_tgt_flavour

  @staticmethod
  def _resolve_mode_and_opt_level(
    toml_config: typing.Mapping[str, JsonValue],
    mode: str | ConversionMode | None,
    enable_graph_optimization: bool | None,
    opt_level: str | OptimizationLevel | None,
  ) -> tuple[ConversionMode, bool, OptimizationLevel]:
    """Resolve conversion mode, graph optimization flag, and optimization level across CLI and TOML.

    Args:
        toml_config: Parsed ``[tool.ml_switcheroo]`` mapping.
        mode: Optional CLI conversion mode override.
        enable_graph_optimization: Optional CLI graph optimization override.
        opt_level: Optional CLI optimization level override.

    Returns:
        Tuple of ``(conversion_mode, enable_graph_optimization, opt_level)``.
    """
    raw_mode = toml_config.get("mode")
    toml_graph_opt = bool(toml_config.get("enable_graph_optimization", False))
    if mode is not None:
      final_mode = ConversionMode(mode)
      final_graph_opt = (
        enable_graph_optimization if enable_graph_optimization is not None else (final_mode == ConversionMode.OPTIMIZED)
      )
    elif enable_graph_optimization is not None:
      final_graph_opt = enable_graph_optimization
      final_mode = ConversionMode.OPTIMIZED if final_graph_opt else ConversionMode.PRESERVE_TRIVIA
    elif isinstance(raw_mode, str) and raw_mode in {m.value for m in ConversionMode}:
      final_mode = ConversionMode(raw_mode)
      final_graph_opt = (final_mode == ConversionMode.OPTIMIZED) or toml_graph_opt
    else:
      final_graph_opt = toml_graph_opt
      final_mode = ConversionMode.OPTIMIZED if final_graph_opt else ConversionMode.PRESERVE_TRIVIA

    raw_opt_level = toml_config.get("opt_level")
    if opt_level is not None:
      final_opt_level = OptimizationLevel(opt_level)
    elif isinstance(raw_opt_level, str) and raw_opt_level in {lvl.value for lvl in OptimizationLevel}:
      final_opt_level = OptimizationLevel(raw_opt_level)
    else:
      final_opt_level = OptimizationLevel.O2

    return final_mode, final_graph_opt, final_opt_level

  @staticmethod
  def _resolve_backend(
    toml_config: typing.Mapping[str, JsonValue],
    execution_backend: str | ExecutionBackend | None,
  ) -> ExecutionBackend:
    """Resolve the active ``ExecutionBackend`` from CLI override or TOML configuration.

    Args:
        toml_config: Parsed ``[tool.ml_switcheroo]`` mapping.
        execution_backend: Optional CLI backend override.

    Returns:
        Resolved ``ExecutionBackend`` enum value.
    """
    if execution_backend is not None:
      return ExecutionBackend(execution_backend)
    raw_backend = toml_config.get("execution_backend")
    if isinstance(raw_backend, str) and raw_backend in {b.value for b in ExecutionBackend}:
      return ExecutionBackend(raw_backend)
    return ExecutionBackend.WEBGPU

  @staticmethod
  def _resolve_plugin_paths(toml_config: typing.Mapping[str, JsonValue], toml_dir: Path | None) -> list[Path]:
    """Resolve ``plugin_paths`` entries from TOML relative to ``toml_dir`` when available.

    Args:
        toml_config: Parsed ``[tool.ml_switcheroo]`` mapping.
        toml_dir: Directory containing the discovered ``pyproject.toml``, if any.

    Returns:
        List of resolved absolute plugin directory ``Path`` instances.
    """
    raw_paths_val = toml_config.get("plugin_paths", [])
    raw_paths: list[str] = (
      [str(p) for p in raw_paths_val if isinstance(p, (str, Path))] if isinstance(raw_paths_val, list) else []
    )
    if toml_dir is not None:
      return [(toml_dir / Path(p)).resolve() for p in raw_paths]
    return [Path(p).resolve() for p in raw_paths]

  @classmethod
  def load(
    cls,
    overrides: "RuntimeConfigOverrides | None" = None,
    *,
    source: str | None = None,
    target: str | None = None,
    strict_mode: bool | None = None,
    search_path: Path | None = None,
  ) -> "RuntimeConfig":
    """Load configuration from ``pyproject.toml``, overriding with typed CLI/API overrides.

    Defaults are calculated dynamically via factory methods if not found.

    Args:
        overrides: Optional ``RuntimeConfigOverrides`` bundle of CLI/API overrides.
        source: Convenience override for source framework.
        target: Convenience override for target framework.
        strict_mode: Convenience override for strict mode setting.
        search_path: Convenience directory path to begin ``pyproject.toml`` discovery.

    Returns:
        RuntimeConfig: The fully resolved configuration object.
    """
    ov = overrides if overrides is not None else RuntimeConfigOverrides()
    eff_source = source if source is not None else ov.source
    eff_target = target if target is not None else ov.target
    eff_strict = strict_mode if strict_mode is not None else ov.strict_mode
    eff_search = search_path if search_path is not None else ov.search_path

    toml_config, toml_dir = _load_toml_settings(eff_search or Path.cwd())
    final_source, final_target, final_src_flavour, final_tgt_flavour = cls._resolve_frameworks_and_flavours(
      toml_config, eff_source, eff_target, ov.source_flavour, ov.target_flavour
    )

    final_strict = eff_strict if eff_strict is not None else bool(toml_config.get("strict_mode", False))
    raw_inter = toml_config.get("intermediate")
    final_intermediate = validate_intermediate_format(
      ov.intermediate if ov.intermediate is not None else (str(raw_inter) if isinstance(raw_inter, str) else None)
    )

    final_mode, final_graph_opt, final_opt_level = cls._resolve_mode_and_opt_level(
      toml_config, ov.mode, ov.enable_graph_optimization, ov.opt_level
    )
    final_sharding = (
      ov.enable_sharding if ov.enable_sharding is not None else bool(toml_config.get("enable_sharding", False))
    )
    final_import_fixer = bool(toml_config.get("enable_import_fixer", True))
    final_backend = cls._resolve_backend(toml_config, ov.execution_backend)
    final_on_device = (
      ov.enable_on_device_execution
      if ov.enable_on_device_execution is not None
      else bool(toml_config.get("enable_on_device_execution", False))
    )
    final_emit_bwd = (
      ov.emit_backward_code if ov.emit_backward_code is not None else bool(toml_config.get("emit_backward_code", False))
    )

    raw_toml_plugins = toml_config.get("plugin_settings", {})
    toml_plugins: dict[str, JsonValue] = dict(raw_toml_plugins) if isinstance(raw_toml_plugins, dict) else {}
    cli_plugins: dict[str, JsonValue] = dict(ov.plugin_settings) if ov.plugin_settings else {}
    final_plugins: dict[str, JsonValue] = {**toml_plugins, **cli_plugins}

    final_report = ov.validation_report
    if not final_report and "validation_report" in toml_config:
      final_report = Path(str(toml_config["validation_report"]))

    final_plugin_paths = cls._resolve_plugin_paths(toml_config, toml_dir)

    return cls(
      source_framework=final_source,
      target_framework=final_target,
      source_flavour=final_src_flavour,
      target_flavour=final_tgt_flavour,
      strict_mode=final_strict,
      intermediate=final_intermediate,
      mode=final_mode,
      opt_level=final_opt_level,
      enable_graph_optimization=final_graph_opt,
      enable_sharding=final_sharding,
      enable_import_fixer=final_import_fixer,
      enable_on_device_execution=final_on_device,
      execution_backend=final_backend,
      emit_backward_code=final_emit_bwd,
      plugin_settings=final_plugins,
      plugin_paths=final_plugin_paths,
      validation_report=final_report,
    )


class RuntimeConfigOverrides(BaseModel):
  """Typed CLI/API override bundle for ``RuntimeConfig.load`` and ``ASTEngine.from_options``.

  Attributes:
      source: Optional source framework override.
      target: Optional target framework override.
      source_flavour: Optional source flavour override.
      target_flavour: Optional target flavour override.
      strict_mode: Optional strict mode override.
      intermediate: Optional intermediate representation mode override.
      enable_graph_optimization: Optional graph optimization override.
      enable_sharding: Optional sharding semantics override.
      plugin_settings: Optional plugin settings mapping.
      validation_report: Optional validation report file path.
      search_path: Optional directory path to begin ``pyproject.toml`` discovery.
      mode: Optional conversion mode override.
      opt_level: Optional optimization level override.
      execution_backend: Optional execution backend override.
      enable_on_device_execution: Optional on-device forward/backward execution override.
      emit_backward_code: Optional explicit backward/VJP synthesis override.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  source: str | None = None
  target: str | None = None
  source_flavour: str | None = None
  target_flavour: str | None = None
  strict_mode: bool | None = None
  intermediate: str | None = None
  enable_graph_optimization: bool | None = None
  enable_sharding: bool | None = None
  plugin_settings: typing.Mapping[str, JsonValue] | None = None
  validation_report: Path | None = None
  search_path: Path | None = None
  mode: str | ConversionMode | None = None
  opt_level: str | OptimizationLevel | None = None
  execution_backend: str | ExecutionBackend | None = None
  enable_on_device_execution: bool | None = None
  emit_backward_code: bool | None = None


def _load_toml_settings(start_path: Path) -> tuple[dict[str, JsonValue], Path | None]:
  """Recursively searches parents for 'pyproject.toml' and extracts config.

  Args:
      start_path (Path): Directory to begin the search from.

  Returns:
      Tuple[Dict[str, JsonValue], Optional[Path]]:
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

        raw_tool = data.get("tool", {})
        tool_section = raw_tool.get("ml_switcheroo", {}) if isinstance(raw_tool, dict) else {}
        return (dict(tool_section) if isinstance(tool_section, dict) else {}), path
      except (OSError, ValueError, TypeError, KeyError):
        # Malformed TOML or permission error implies skip
        return {}, None

  return {}, None


def parse_cli_key_values(
  items: list[str] | None,
) -> dict[str, str | int | float | bool]:
  """Parse a list of 'key=value' strings into a dictionary with type inference.

  Args:
      items (Optional[List[str]]): List of strings in "key=value" format.

  Returns:
      Dict[str, Union[str, int, float, bool]]: Dictionary with parsed values (bool, int, float, or str).

  """
  if not items:
    return {}

  config: dict[str, str | int | float | bool] = {}
  for item in items:
    if "=" not in item:
      print(f"⚠️  Ignoring invalid config format: '{item}'. Expected 'key=value'.")
      continue

    key, val_str = item.split("=", 1)
    key = key.strip()
    val_str = val_str.strip()

    final_val: str | int | float | bool = val_str

    # Basic type inference
    lower_val = val_str.lower()
    if lower_val == "true":
      final_val = True
    elif lower_val == "false":
      final_val = False
    else:
      with contextlib.suppress(ValueError):
        final_val = float(val_str) if "." in val_str or "e" in lower_val else int(val_str)

    config[key] = final_val

  return config
