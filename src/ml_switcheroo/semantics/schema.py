"""
Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`k_array_api.json`, etc.)
and the Framework Configuration blocks.

Updated to support:
- `PluginTraits`: Capability flags for decoupling plugins from framework identifiers.
- Rich Argument Definitions in `std_args`.
- Conditional Dispatch Rules.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Set
from pydantic import BaseModel, Field, ConfigDict

from ml_switcheroo.enums import SemanticTier, LogicOp


class PluginTraits(BaseModel):
  """
  Defines capability flags for a framework to guide plugin logic.
  This decouples plugins from hardcoded framework names (e.g., checking "if jax"
  becomes "if traits.has_numpy_compatible_arrays").
  """

  model_config = ConfigDict(extra="allow")

  # --- Array Capabilities ---
  # Does the framework support numpy-style array creation, iteration, and dtypes?
  # Used by: Casting plugin (.astype), Padding plugin (tuple-of-tuples), etc.
  # True for: JAX, NumPy, TensorFlow, MLX, Keras (backend-agnostic)
  # False for: PyTorch (uses .to(), specific padding tuples)
  has_numpy_compatible_arrays: bool = Field(
    False,
    description="If True, supports .astype(), numpy-style padding, and array properties.",
  )

  # --- RNG Semantics ---
  requires_explicit_rng: bool = Field(
    False,
    description="If True, stochastic operations require threaded PRNG keys (JAX-style).",
  )

  # --- State Semantics ---
  requires_functional_state: bool = Field(
    False,
    description="If True, stateful layers (BN/Optimizers) return updated state explicitly.",
  )

  # --- Functional Purity ---
  # Does the framework require defining loops via functional primitives (scan/fori_loop)?
  # Used by: Loop Unrolling plugin.
  # True for: JAX
  # False for: PyTorch, NumPy, TensorFlow (Eager), MLX
  requires_functional_control_flow: bool = Field(
    False,
    description="If True, standard Python loops are unsafe for JIT/Graph mode.",
  )

  # --- Safety Checks ---
  # Does the framework generally forbid side-effects (IO, Globals, Mutations) due to JIT/XLA?
  # Used by: ASTEngine (PurityScanner trigger).
  # True for: JAX, Flax, PaxML
  enforce_purity_analysis: bool = Field(
    False,
    description="If True, the Engine runs PurityScanner to detect side-effects (globals, IO) before transpilation.",
  )


class StructuralTraits(BaseModel):
  """
  Defines structural patterns for a framework to guide generic rewriters.
  This allows adding new frameworks (like MLX, TinyGrad) without editing python code.
  """

  model_config = ConfigDict(extra="allow")

  # --- Class Inheritance ---
  # e.g. "flax.nnx.Module" or "torch.nn.Module"
  module_base: Optional[str] = Field(None, description="Base class API for neural modules.")

  # --- Method Renaming ---
  # e.g. "__call__" (JAX/Flax) vs "forward" (PyTorch) vs "call" (Keras)
  forward_method: Optional[str] = Field(None, description="Standard method name for forward pass.")

  # --- Known Inference Methods (New) ---
  # List of method names that are treated as 'forward/inference' methods across frameworks.
  # This allows detecting 'predict', 'run', etc. in custom frameworks.
  # Default covers standard libs.
  known_inference_methods: Set[str] = Field(
    default={"forward", "__call__", "call"},
    description="Set of method names recognized as model inference entry points.",
  )

  # --- Constructor Logic ---
  requires_super_init: bool = Field(False, description="If True, injects super().__init__() in constructors.")
  init_method_name: Optional[str] = Field("__init__", description="Name of the constructor (e.g. 'setup' for Pax).")

  # --- Argument Injection/Stripping (Magic State) ---
  # e.g. [("rngs", "nnx.Rngs")]
  inject_magic_args: List[Tuple[str, Optional[str]]] = Field(
    default_factory=list, description="List of (name, type) tuples to inject into signatures."
  )
  # e.g. ["rngs", "key"]
  strip_magic_args: List[str] = Field(
    default_factory=list, description="List of argument names to remove from signatures."
  )

  # --- Lifecycle Method Handling ---
  # Methods to silently remove from call chains (e.g. .cuda(), .detach())
  lifecycle_strip_methods: List[str] = Field(
    default_factory=list, description="List of method names to strip from chains (identity transform)."
  )
  # Methods to remove but warn about (e.g. .eval(), .train())
  lifecycle_warn_methods: List[str] = Field(
    default_factory=list, description="List of method names to strip but trigger a warning."
  )

  # --- Analysis: Impurity Detection ---
  # Framework-Specific methods that mutate state in place (violating functional purity)
  # e.g. ["add_", "copy_", "zero_"]
  impurity_methods: List[str] = Field(default_factory=list, description="List of method names considered side-effects.")

  # --- JIT Configuration ---
  # Methods argument names that must be static during JIT compilation
  # e.g. ["axis", "dims", "keepdims"]
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

  # Feature: Capability Tiers (Added to support safety checks)
  tiers: Optional[List[SemanticTier]] = Field(
    default=None, description="List of Semantic Tiers (array, neural, extras) this framework supports."
  )

  # Feature: Structural Traits
  traits: Optional[StructuralTraits] = Field(
    default_factory=StructuralTraits, description="Configuration for AST structural rewriting."
  )

  # Feature: Plugin Traits (New for Decoupling)
  plugin_traits: Optional[PluginTraits] = Field(
    default_factory=PluginTraits, description="Configuration logic flags for plugins."
  )


class Rule(BaseModel):
  """
  Declarative rule for conditional logic within a variant.
  Evaluated at runtime to dynamically switch APIs.
  """

  model_config = ConfigDict(populate_by_name=True)

  if_arg: str = Field(..., description="Name of the standard argument to check.")
  op: LogicOp = Field(LogicOp.EQ, description="Logical operator for comparison.")
  is_val: Union[str, int, float, bool, List[Union[str, int, float]]] = Field(
    ..., alias="val", description="Value or list of values to compare against."
  )
  use_api: str = Field(..., description="The target API path to use if the condition matches.")


class Variant(BaseModel):
  """
  Defines how a specific framework implements an abstract operation.
  """

  model_config = ConfigDict(extra="allow")

  api: Optional[str] = Field(None, description="Fully qualified API path (e.g. 'jax.numpy.sum').")
  args: Optional[Dict[str, str]] = Field(None, description="Map of 'Standard Name' -> 'Framework Name' for arguments.")
  arg_values: Optional[Dict[str, Dict[str, str]]] = Field(
    None,
    description="Map of {StandardArg: {SourceValueString: TargetValueCode}} for enum value mapping.",
  )
  kwargs_map: Optional[Dict[str, str]] = Field(None, description="Mapping for specific keys within a **kwargs expansion.")
  inject_args: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
    None,
    description="Dictionary of new arguments to inject with fixed default values (e.g. {'epsilon': 1e-5}).",
  )
  casts: Optional[Dict[str, str]] = Field(
    None, description="Mapping of argument names to target types (e.g. {'x': 'int32'})."
  )
  requires_plugin: Optional[str] = Field(None, description="Name of a registered plugin hook to handle translation.")

  # --- Feature: Conditional Dispatch ---
  dispatch_rules: List[Rule] = Field(
    default_factory=list,
    description="List of conditional rules to switch APIs based on argument values at runtime.",
  )

  # --- Feature: Argument Packing ---
  pack_to_tuple: Optional[str] = Field(
    None,
    description="If set (e.g. 'axes'), collects variadic positional args into a tuple kwargs.",
  )

  # --- Feature: Tensor Layout Permutation ---
  layout_map: Optional[Dict[str, str]] = Field(
    None,
    description="Map of arguments to layout transformation strings (e.g., {'x': 'NCHW->NHWC'}).",
  )

  transformation_type: Optional[str] = Field(None, description="Special rewrite mode (e.g. 'infix', 'inline_lambda').")
  operator: Optional[str] = Field(None, description="If transformation_type='infix', the symbol to use (e.g. '+').")

  # --- Feature: Composite Operations (Macros) ---
  macro_template: Optional[str] = Field(
    None,
    description="Expression template for composite ops (e.g. '{x} * functional.sigmoid({x})').",
  )

  output_select_index: Optional[int] = Field(
    None,
    description="Integer index to extract from a tuple return value.",
  )
  output_adapter: Optional[str] = Field(
    None,
    description=(
      "Python lambda string to normalize the return value. Wraps the target call result. Example: 'lambda x: x[0]'."
    ),
  )
  output_cast: Optional[str] = Field(
    None,
    description="Target Dtype string (e.g. 'jnp.int64') to cast the output to via .astype().",
  )

  min_version: Optional[str] = Field(None, description="Minimum supported version.")
  max_version: Optional[str] = Field(None, description="Maximum supported version.")

  # --- Feature: Dynamic Imports ---
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
  default: Optional[str] = Field(None, description="Default value as a string code representation.")
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

  operation: Optional[str] = None  # Backcompat
  description: Optional[str] = None
  from_file: Optional[str] = Field(None, alias="from")

  # std_args can be simple ["x"], typed [("x", "int")], or rich Dicts [{"name": "x", "min": 0}]
  std_args: List[Union[str, Tuple[str, str], ParameterDef, Dict[str, Any]]] = Field(
    default_factory=list, description="List of standard argument names, types, or rich definitions."
  )

  variants: Dict[str, Optional[Variant]] = Field(
    default_factory=dict, description="Map of framework keys ('torch', 'jax') to implementation details."
  )

  # --- Feature: Scaffold Plugins / Verification ---
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

  # Meta blocks
  frameworks: Optional[Dict[str, FrameworkTraits]] = Field(None, alias="__frameworks__")
  imports: Optional[Dict[str, Any]] = Field(None, alias="__imports__")

  # Operations map (everything else)
  pass
