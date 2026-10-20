"""Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`odl/*.yaml`, etc.)
and the Framework Configuration blocks.

Updated to support:
- `PluginTraits`: Capability flags for decoupling plugins from framework identifiers.
- Rich Argument Definitions in `std_args` with Typed Defaults (Any).
- Conditional Dispatch Rules.
- **PatternDef**: Graph fusion patterns.
"""

import typing

from ml_switcheroo_ir.schema.ghost import SemanticTier as SemanticTier
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ml_switcheroo.core.dsl import PatternDef as PatternDef


class VariantMappingDict(BaseModel):
  """Pydantic model representing a raw framework variant mapping in the Semantic Knowledge Base."""

  model_config = ConfigDict(extra="forbid")

  api: typing.Optional[str] = None
  args: typing.Optional[typing.Dict[str, typing.Optional[str]]] = None
  kwargs_map: typing.Optional[typing.Dict[str, typing.Optional[str]]] = None
  arg_values: typing.Optional[typing.Dict[str, typing.Dict[str, JsonValue]]] = None
  requires_plugin: typing.Optional[str] = None
  macro_template: typing.Optional[str] = None
  transformation_type: typing.Optional[str] = None
  operator: typing.Optional[str] = None
  pack_to_tuple: typing.Optional[str] = None
  pack_as: typing.Optional[str] = None
  imports: typing.Optional[typing.List[typing.Union[str, typing.Dict[str, typing.Optional[str]]]]] = None
  min_version: typing.Optional[str] = None
  max_version: typing.Optional[str] = None
  replace_With_graph: typing.Optional[typing.List[typing.Dict[str, JsonValue]]] = None
  output_select: typing.Optional[int] = None
  output_cast: typing.Optional[str] = None
  inject_args: typing.Optional[typing.Dict[str, JsonValue]] = None
  layout_map: typing.Optional[typing.Dict[str, str]] = None
  rule: typing.Optional[typing.List[typing.Dict[str, JsonValue]]] = None


class ParameterDefDict(BaseModel):
  """Pydantic model representing a raw parameter definition in std_args."""

  model_config = ConfigDict(extra="forbid")

  name: str = ""
  type: typing.Optional[str] = None
  doc: typing.Optional[str] = None
  default: JsonValue = None
  min: typing.Optional[float] = None
  max: typing.Optional[float] = None
  options: typing.Optional[typing.List[typing.Union[str, int, float, bool]]] = None
  rank: typing.Optional[int] = None
  dtype: typing.Optional[str] = None
  shape_spec: typing.Optional[str] = None
  is_variadic: bool = False
  kind: str = "positional_or_keyword"


class OperationDataDict(BaseModel):
  """Pydantic model representing a raw ODL operation entry in the Semantic Knowledge Base."""

  model_config = ConfigDict(extra="forbid")

  operation: str = ""
  description: str = ""
  op_type: str = "function"
  std_args: typing.List[typing.Union[str, typing.Tuple[str, str], ParameterDefDict]] = Field(default_factory=list)
  variants: typing.Dict[str, typing.Optional[VariantMappingDict]] = Field(default_factory=dict)
  return_type: str = "Any"
  is_inplace: bool = False
  output_shape_calc: typing.Optional[str] = None
  sharding_supported: bool = False
  tensor_rank_Supported: typing.Optional[typing.List[int]] = None
  doc_url: typing.Optional[str] = None
  scaffold_plugins: typing.List[typing.Dict[str, JsonValue]] = Field(default_factory=list)
  extras: typing.Dict[str, JsonValue] = Field(default_factory=dict)
  is_internal: bool = Field(default=False, alias="_is_internal")


class FrameworkConfigDict(BaseModel):
  """Pydantic model representing a raw framework configuration dictionary in SemanticsManager."""

  model_config = ConfigDict(extra="forbid")

  extends: typing.Optional[str] = None
  alias: typing.Dict[str, str] = Field(default_factory=dict)
  stateful_call: typing.Dict[str, str] = Field(default_factory=dict)
  tiers: typing.List[str] = Field(default_factory=list)
  traits: typing.Dict[str, JsonValue] = Field(default_factory=dict)
  plugin_traits: typing.Dict[str, JsonValue] = Field(default_factory=dict)


class ProviderConfigDict(BaseModel):
  """Pydantic model representing a provider namespace configuration in SemanticsManager."""

  model_config = ConfigDict(extra="forbid")

  root: str = ""
  sub: typing.Optional[str] = None
  alias: typing.Optional[str] = None


VariantDict = VariantMappingDict
VariantMappingModel = VariantMappingDict
ParameterDefModel = ParameterDefDict
OperationDataModel = OperationDataDict
FrameworkConfigModel = FrameworkConfigDict
ProviderConfigModel = ProviderConfigDict


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
  strict_materialization_method: typing.Optional[str] = Field(
    default=None, description="Method name to force materialization in strict mode (e.g. 'block_until_ready')."
  )
  sharding_wrapper_api: typing.Optional[str] = Field(
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

  module_base: typing.Optional[str] = Field(default=None, description="Base class API for neural modules.")
  tensor_type: typing.Optional[str] = Field(
    default=None, description="Canonical tensor type annotation for this framework."
  )
  forward_method: typing.Optional[str] = Field(default=None, description="Standard method name for forward pass.")
  known_inference_methods: typing.Set[str] = Field(
    default={"forward", "__call__", "call"},
    description="Set of method names recognized as model inference entry points.",
  )
  functional_execution_method: typing.Optional[str] = Field(
    default=None,
    description="Method name used for functional execution (e.g. 'apply'). Rewriter unwraps this pattern.",
  )
  requires_super_init: bool = Field(default=False, description="If True, injects super().__init__() in constructors.")
  init_method_name: typing.Optional[str] = Field(
    default="__init__", description="Name of the constructor (e.g. 'setup' for Pax)."
  )
  inject_magic_args: typing.List[typing.Tuple[str, typing.Optional[str]]] = Field(
    default_factory=list, description="List of (name, type) tuples to inject into signatures."
  )
  strip_magic_args: typing.List[str] = Field(
    default_factory=list, description="Explicit list of argument names to remove from signatures."
  )
  auto_strip_magic_args: bool = Field(
    default=False,
    description="If True, automatically strips all magic arguments (built from the global registry) not used by this framework.",
  )
  lifecycle_strip_methods: typing.List[str] = Field(
    default_factory=list, description="List of method names to strip from chains (identity transform)."
  )
  lifecycle_warn_methods: typing.List[str] = Field(
    default_factory=list, description="List of method names to strip but trigger a warning."
  )
  impurity_methods: typing.List[str] = Field(
    default_factory=list, description="List of method names considered side-effects."
  )
  implicit_method_roots: typing.List[str] = Field(
    default_factory=list, description="List of fully qualified class names to try as prefixes for method calls."
  )
  jit_static_args: typing.List[str] = Field(
    default_factory=list, description="Argument keywords that require static compilation."
  )


class FrameworkAlias(BaseModel):
  """Configuration for global module aliasing."""

  module: str
  name: str


class FrameworkTraits(BaseModel):
  """Configuration for a framework's structural behavior."""

  model_config = ConfigDict(extra="allow")

  extends: typing.Optional[str] = Field(
    default=None, description="Base framework to inherit variants from (e.g. paxml extends jax)."
  )
  alias: typing.Optional[FrameworkAlias] = Field(default=None, description="Default import alias configuration.")
  stateful_call: typing.Optional[typing.Dict[str, str]] = Field(
    None, description="Configuration for stateful calling conventions."
  )
  tiers: typing.Optional[typing.List[SemanticTier]] = Field(
    default=None, description="List of Semantic Tiers (array, neural, extras) this framework supports."
  )
  traits: typing.Optional[StructuralTraits] = Field(
    default_factory=StructuralTraits, description="Configuration for AST structural rewriting."
  )
  plugin_traits: typing.Optional[PluginTraits] = Field(
    default_factory=PluginTraits, description="Configuration logic flags for plugins."
  )


class SemanticsFile(BaseModel):
  """Schema representing an entire semantics JSON file."""

  model_config = ConfigDict(extra="allow")
  frameworks: typing.Optional[typing.Dict[str, FrameworkTraits]] = Field(None, alias="__frameworks__")
  imports: typing.Optional[typing.Dict[str, JsonValue]] = Field(None, alias="__imports__")
  patterns: typing.Optional[typing.List[PatternDef]] = Field(default_factory=lambda: [], alias="__patterns__")


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
