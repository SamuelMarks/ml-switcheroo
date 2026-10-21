"""Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`odl/*.yaml`, etc.)
and the Framework Configuration blocks.

Updated to support:
- `PluginTraits`: Capability flags for decoupling plugins from framework identifiers.
- Rich Argument Definitions in `std_args` with Typed Defaults (Any).
- Conditional Dispatch Rules.
- **PatternDef**: Graph fusion patterns.
"""

from ml_switcheroo_ir.schema.ghost import SemanticTier as SemanticTier
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ml_switcheroo.core.dsl import PatternDef as PatternDef


class VariantMappingModel(BaseModel):
  """Pydantic model representing a raw framework variant mapping in the Semantic Knowledge Base."""

  model_config = ConfigDict(extra="forbid")

  api: str | None = None
  args: dict[str, str | None] | None = None
  kwargs_map: dict[str, str | None] | None = None
  arg_values: dict[str, dict[str, JsonValue]] | None = None
  requires_plugin: str | None = None
  macro_template: str | None = None
  transformation_type: str | None = None
  operator: str | None = None
  pack_to_tuple: str | None = None
  pack_as: str | None = None
  imports: list[str | dict[str, str | None]] | None = None
  min_version: str | None = None
  max_version: str | None = None
  replace_With_graph: list[dict[str, JsonValue]] | None = None
  output_select: int | None = None
  output_cast: str | None = None
  inject_args: dict[str, JsonValue] | None = None
  layout_map: dict[str, str] | None = None
  rule: list[dict[str, JsonValue]] | None = None


class ParameterDefModel(BaseModel):
  """Pydantic model representing a raw parameter definition in std_args."""

  model_config = ConfigDict(extra="forbid")

  name: str = ""
  type: str | None = None
  doc: str | None = None
  default: JsonValue = None
  min: float | None = None
  max: float | None = None
  options: list[str | int | float | bool] | None = None
  rank: int | None = None
  dtype: str | None = None
  shape_spec: str | None = None
  is_variadic: bool = False
  kind: str = "positional_or_keyword"


class OperationDataModel(BaseModel):
  """Pydantic model representing a raw ODL operation entry in the Semantic Knowledge Base."""

  model_config = ConfigDict(extra="forbid")

  operation: str = ""
  description: str = ""
  op_type: str = "function"
  std_args: list[str | tuple[str, str] | ParameterDefModel] = Field(default_factory=list)
  variants: dict[str, VariantMappingModel | None] = Field(default_factory=dict)
  return_type: str = "Any"
  is_inplace: bool = False
  output_shape_calc: str | None = None
  sharding_supported: bool = False
  tensor_rank_Supported: list[int] | None = None
  doc_url: str | None = None
  scaffold_plugins: list[dict[str, JsonValue]] = Field(default_factory=list)
  extras: dict[str, JsonValue] = Field(default_factory=dict)
  is_internal: bool = Field(default=False, alias="_is_internal")


class FrameworkConfigModel(BaseModel):
  """Pydantic model representing a raw framework configuration dictionary in SemanticsManager."""

  model_config = ConfigDict(extra="forbid")

  extends: str | None = None
  alias: dict[str, str] = Field(default_factory=dict)
  stateful_call: dict[str, str] = Field(default_factory=dict)
  tiers: list[str] = Field(default_factory=list)
  traits: dict[str, JsonValue] = Field(default_factory=dict)
  plugin_traits: dict[str, JsonValue] = Field(default_factory=dict)


class ProviderConfigModel(BaseModel):
  """Pydantic model representing a provider namespace configuration in SemanticsManager."""

  model_config = ConfigDict(extra="forbid")

  root: str = ""
  sub: str | None = None
  alias: str | None = None


class PluginTraits(BaseModel):
  """Define capability flags for a framework to guide plugin logic."""

  model_config = ConfigDict(extra="allow")

  has_numpy_compatible_arrays: bool = Field(
    False,
    description="If True, supports .astype(), NumPy-style padding, and array properties.",
  )
  requires_explicit_rng: bool = Field(
    False,
    description="If True, stochastic operations require threaded PRNG keys (JAX-style).",
  )
  requires_functional_state: bool = Field(
    default=False,
    description="If True, stateful layers (BN/Optimizers) return updated state explicitly.",
  )
  requires_functional_control_flow: bool = Field(
    False,
    description="If True, standard Python loops are unsafe for JIT/Graph mode.",
  )
  enforce_purity_analysis: bool = Field(
    default=False,
    description="If True, the Engine runs PurityScanner to detect side-effects (globals, IO) before transpilation.",
  )
  strict_materialization_method: str | None = Field(
    default=None, description="Method name to force materialization in strict mode (e.g. 'block_until_ready')."
  )
  sharding_wrapper_api: str | None = Field(
    default=None,
    description="Fully qualified API for wrapping operations in sharding (e.g. 'torch.distributed.fsdp.FSDP' or 'jax.experimental.pjit.pjit').",
  )
  supports_dynamic_shapes: bool = Field(
    default=False,
    description="If True, the framework/IR supports dynamic symbolic tensor dimensions.",
  )
  supports_sharding: bool = Field(
    default=False,
    description="If True, the framework/IR supports distributed sharding annotations.",
  )


class StructuralTraits(BaseModel):
  """Define structural patterns for a framework to guide generic rewriters."""

  model_config = ConfigDict(extra="allow")

  module_base: str | None = Field(default=None, description="Base class API for neural modules.")
  tensor_type: str | None = Field(default=None, description="Canonical tensor type annotation for this framework.")
  forward_method: str | None = Field(default=None, description="Standard method name for forward pass.")
  known_inference_methods: set[str] = Field(
    default={"forward", "__call__", "call"},
    description="Set of method names recognized as model inference entry points.",
  )
  functional_execution_method: str | None = Field(
    default=None,
    description="Method name used for functional execution (e.g. 'apply'). Rewriter unwraps this pattern.",
  )
  requires_super_init: bool = Field(default=False, description="If True, injects super().__init__() in constructors.")
  init_method_name: str | None = Field(default="__init__", description="Name of the constructor (e.g. 'setup' for Pax).")
  inject_magic_args: list[tuple[str, str | None]] = Field(
    default_factory=list, description="List of (name, type) tuples to inject into signatures."
  )
  strip_magic_args: list[str] = Field(
    default_factory=list, description="Explicit list of argument names to remove from signatures."
  )
  auto_strip_magic_args: bool = Field(
    default=False,
    description="If True, automatically strips all magic arguments (built from the global registry) not used by this framework.",
  )
  lifecycle_strip_methods: list[str] = Field(
    default_factory=list, description="List of method names to strip from chains (identity transform)."
  )
  lifecycle_warn_methods: list[str] = Field(
    default_factory=list, description="List of method names to strip but trigger a warning."
  )
  impurity_methods: list[str] = Field(default_factory=list, description="List of method names considered side-effects.")
  implicit_method_roots: list[str] = Field(
    default_factory=list, description="List of fully qualified class names to try as prefixes for method calls."
  )
  jit_static_args: list[str] = Field(
    default_factory=list, description="Argument keywords that require static compilation."
  )


class FrameworkAlias(BaseModel):
  """Configuration for global module aliasing."""

  module: str
  name: str


class FrameworkTraits(BaseModel):
  """Configuration for a framework's structural behavior."""

  model_config = ConfigDict(extra="allow")

  extends: str | None = Field(
    default=None, description="Base framework to inherit variants from (e.g. paxml extends jax)."
  )
  alias: FrameworkAlias | None = Field(default=None, description="Default import alias configuration.")
  stateful_call: dict[str, str] | None = Field(None, description="Configuration for stateful calling conventions.")
  tiers: list[SemanticTier] | None = Field(
    default=None, description="List of Semantic Tiers (array, neural, extras) this framework supports."
  )
  traits: StructuralTraits | None = Field(
    default_factory=StructuralTraits, description="Configuration for AST structural rewriting."
  )
  plugin_traits: PluginTraits | None = Field(
    default_factory=PluginTraits, description="Configuration logic flags for plugins."
  )


class SemanticsFile(BaseModel):
  """Schema representing an entire semantics JSON file."""

  model_config = ConfigDict(extra="allow")
  frameworks: dict[str, FrameworkTraits] | None = Field(None, alias="__frameworks__")
  imports: dict[str, JsonValue] | None = Field(None, alias="__imports__")
  patterns: list[PatternDef] | None = Field(default_factory=lambda: [], alias="__patterns__")


def validate_yaml_schema(yaml_content: str) -> SemanticsFile:
  """Validates a YAML string against the SemanticsFile schema.

  Args:
      yaml_content: The YAML string to validate.

  Returns:
      A validated SemanticsFile instance.

  Raises:
      ValueError: If the YAML content is invalid or fails schema validation.
  """
  import yaml
  from pydantic import ValidationError

  try:
    data = yaml.safe_load(yaml_content)
  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML content: {e}")

  if data is None:
    data = {}

  try:
    return SemanticsFile.model_validate(data)
  except ValidationError as e:
    raise ValueError(f"Schema validation failed: {e}")
