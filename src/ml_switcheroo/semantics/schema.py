"""
Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`k_array_api.json`, etc.)
and the Framework Configuration blocks.

Updated to support:
- `PluginTraits`: Capability flags for decoupling plugins from framework identifiers.
- Rich Argument Definitions in `std_args` with Typed Defaults (Any).
- Conditional Dispatch Rules.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Set, Literal
from pydantic import BaseModel, Field, ConfigDict

from ml_switcheroo.enums import SemanticTier, LogicOp


class PluginTraits(BaseModel):
  """
  Defines capability flags for a framework to guide plugin logic.
  """

  model_config = ConfigDict(extra="allow")

  has_numpy_compatible_arrays: bool = Field(
    False,
    description="If True, supports .astype(), numpy-style padding, and array properties.",
  )
  requires_explicit_rng: bool = Field(
    False,
    description="If True, stochastic operations require threaded PRNG keys (JAX-style).",
  )
  requires_functional_state: bool = Field(
    False,
    description="If True, stateful layers (BN/Optimizers) return updated state explicitly.",
  )
  requires_functional_control_flow: bool = Field(
    False,
    description="If True, standard Python loops are unsafe for JIT/Graph mode.",
  )
  enforce_purity_analysis: bool = Field(
    False,
    description="If True, the Engine runs PurityScanner to detect side-effects (globals, IO) before transpilation.",
  )
  strict_materialization_method: Optional[str] = Field(
    None, description="Method name to force materialization in strict mode (e.g. 'block_until_ready')."
  )


class StructuralTraits(BaseModel):
  """
  Defines structural patterns for a framework to guide generic rewriters.
  """

  model_config = ConfigDict(extra="allow")

  module_base: Optional[str] = Field(None, description="Base class API for neural modules.")
  forward_method: Optional[str] = Field(None, description="Standard method name for forward pass.")
  known_inference_methods: Set[str] = Field(
    default={"forward", "__call__", "call"},
    description="Set of method names recognized as model inference entry points.",
  )
  functional_execution_method: Optional[str] = Field(
    "apply",
    description="Method name used for functional execution (e.g. 'apply'). Rewriter unwraps this pattern.",
  )
  requires_super_init: bool = Field(False, description="If True, injects super().__init__() in constructors.")
  init_method_name: Optional[str] = Field("__init__", description="Name of the constructor (e.g. 'setup' for Pax).")
  inject_magic_args: List[Tuple[str, Optional[str]]] = Field(
    default_factory=list, description="List of (name, type) tuples to inject into signatures."
  )
  strip_magic_args: List[str] = Field(
    default_factory=list, description="Explicit list of argument names to remove from signatures."
  )
  auto_strip_magic_args: bool = Field(
    False,
    description="If True, automatically strips all magic arguments (built from the global registry) not used by this framework.",
  )
  lifecycle_strip_methods: List[str] = Field(
    default_factory=list, description="List of method names to strip from chains (identity transform)."
  )
  lifecycle_warn_methods: List[str] = Field(
    default_factory=list, description="List of method names to strip but trigger a warning."
  )
  impurity_methods: List[str] = Field(default_factory=list, description="List of method names considered side-effects.")
  implicit_method_roots: List[str] = Field(
    default_factory=list, description="List of fully qualified class names to try as prefixes for method calls."
  )
  jit_static_args: List[str] = Field(
    default_factory=list, description="Argument keywords that require static compilation."
  )


class FrameworkAlias(BaseModel):
  """Configuration for global module aliasing."""

  module: str
  name: str


class FrameworkTraits(BaseModel):
  """
  Configuration for a framework's structural behavior.
  """

  model_config = ConfigDict(extra="allow")

  extends: Optional[str] = Field(None, description="Base framework to inherit variants from (e.g. paxml extends jax).")
  alias: Optional[FrameworkAlias] = Field(None, description="Default import alias configuration.")
  stateful_call: Optional[Dict[str, str]] = Field(None, description="Configuration for stateful calling conventions.")
  tiers: Optional[List[SemanticTier]] = Field(
    default=None, description="List of Semantic Tiers (array, neural, extras) this framework supports."
  )
  traits: Optional[StructuralTraits] = Field(
    default_factory=StructuralTraits, description="Configuration for AST structural rewriting."
  )
  plugin_traits: Optional[PluginTraits] = Field(
    default_factory=PluginTraits, description="Configuration logic flags for plugins."
  )


class Rule(BaseModel):
  """
  Declarative rule for conditional logic within a variant.
  """

  model_config = ConfigDict(populate_by_name=True)

  if_arg: str = Field(..., description="Name of the standard argument to check.")
  op: LogicOp = Field(LogicOp.EQ, description="Logical operator for comparison.")
  is_val: Any = Field(..., alias="val", description="Value or list of values to compare against.")
  use_api: str = Field(..., description="The target API path to use if the condition matches.")


class Variant(BaseModel):
  """
  Defines how a specific framework implements an abstract operation.
  """

  model_config = ConfigDict(extra="allow")

  api: Optional[str] = Field(None, description="Fully qualified API path.")
  args: Optional[Dict[str, str]] = Field(None, description="Map of 'Standard Name' -> 'Framework Name' for arguments.")
  arg_values: Optional[Dict[str, Dict[str, str]]] = Field(
    None,
    description="Map of {StandardArg: {SourceValueString: TargetValueCode}} for enum value mapping.",
  )
  kwargs_map: Optional[Dict[str, str]] = Field(None, description="Mapping for specific keys within a **kwargs expansion.")
  inject_args: Optional[Dict[str, Any]] = Field(
    None,
    description="Dictionary of new arguments to inject with fixed values (supports primitives/collections).",
  )
  casts: Optional[Dict[str, str]] = Field(
    None, description="Mapping of argument names to target types (e.g. {'x': 'int32'})."
  )
  requires_plugin: Optional[str] = Field(None, description="Name of a registered plugin hook to handle translation.")
  dispatch_rules: List[Rule] = Field(
    default_factory=list,
    description="List of conditional rules to switch APIs based on argument values at runtime.",
  )
  pack_to_tuple: Optional[str] = Field(
    None,
    description="If set (e.g. 'axes'), collects variadic positional args into a tuple kwargs.",
  )
  layout_map: Optional[Dict[str, str]] = Field(
    None,
    description="Map of arguments to layout transformation strings (e.g., {'x': 'NCHW->NHWC'}).",
  )
  transformation_type: Optional[str] = Field(None, description="Special rewrite mode (e.g. 'infix', 'inline_lambda').")
  operator: Optional[str] = Field(None, description="If transformation_type='infix', the symbol to use (e.g. '+').")
  macro_template: Optional[str] = Field(
    None,
    description="Expression template for composite ops.",
  )
  output_select_index: Optional[int] = Field(
    None,
    description="Integer index to extract from a tuple return value.",
  )
  output_adapter: Optional[str] = Field(
    None,
    description="Python lambda string to normalize the return value.",
  )
  output_cast: Optional[str] = Field(
    None,
    description="Target Dtype string (e.g. 'jnp.int64') to cast the output to via .astype().",
  )
  min_version: Optional[str] = Field(None, description="Minimum supported version.")
  max_version: Optional[str] = Field(None, description="Maximum supported version.")
  required_imports: List[Union[str, Any]] = Field(
    default_factory=list,
    description="List of imports required by this variant.",
  )
  missing_message: Optional[str] = Field(
    None, description="Custom error message to display if this mapping fails or is missing."
  )


class ParameterDef(BaseModel):
  """
  Definition of a single argument in an abstract operation signature.
  """

  name: str = Field(..., description="Argument name (e.g. 'dim').")
  type: Optional[str] = Field("Any", description="Type hint string (e.g. 'int', 'Tensor').")
  doc: Optional[str] = Field(None, description="Argument docstring explanation.")
  default: Optional[Any] = Field(None, description="Default value (primitive or container).")
  min: Optional[float] = Field(None, description="Minimum numeric value.")
  max: Optional[float] = Field(None, description="Maximum numeric value.")
  options: Optional[List[Union[str, int, float, bool]]] = Field(None, description="List of allowed discrete values.")
  rank: Optional[int] = Field(None, description="Required tensor rank.")
  dtype: Optional[str] = Field(None, description="Required numpy-style dtype.")
  shape_spec: Optional[str] = Field(None, description="Symbolic shape string.")
  is_variadic: bool = Field(False, description="If True, accepts variable args (*args).")
  kind: str = Field("positional_or_keyword", description="Parameter kind.")


class OpDefinition(BaseModel):
  """
  Definition of an Abstract Operation standard.
  """

  model_config = ConfigDict(extra="allow")

  operation: Optional[str] = None
  description: Optional[str] = None

  # std_args can be simple ["x"], typed [("x", "int")], or rich Dicts
  std_args: List[Union[str, Tuple[str, str], ParameterDef, Dict[str, Any]]] = Field(
    default_factory=list, description="List of standard argument names, types, or rich definitions."
  )

  variants: Dict[str, Optional[Variant]] = Field(
    default_factory=dict, description="Map of framework keys to implementation details."
  )

  scaffold_plugins: List[Any] = Field(default_factory=list)
  op_type: Any = "function"
  return_type: Optional[str] = "Any"
  is_inplace: bool = False
  test_rtol: float = 1e-3
  test_atol: float = 1e-4
  skip_fuzzing: bool = False
  nondeterministic: bool = False
  differentiable: bool = True
  deprecated: bool = False
  output_shape_calc: Optional[str] = None
  complexity: Optional[str] = None


class SemanticsFile(BaseModel):
  """Schema representing an entire semantics JSON file."""

  model_config = ConfigDict(extra="allow")
  frameworks: Optional[Dict[str, FrameworkTraits]] = Field(None, alias="__frameworks__")
  imports: Optional[Dict[str, Any]] = Field(None, alias="__imports__")
  pass
