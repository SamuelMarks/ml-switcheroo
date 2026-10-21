"""Operation Definition Language (ODL) Schema.

This module defines the Pydantic models used to validate the input (e.g. YAML)
for defining new operations. It represents the source of truth for generating
new operation definitions in the codebase, enforcing schema validity before
AST injection occurs.

It defines the structure for:

- Parameters (types, constraints, shapes, dtypes).
- Variants (framework implementations).
- Plugins (scaffolding logic).
- Operations (abstract definitions).
- **Patterns**: Graph fusion rules.
"""

from enum import Enum
import typing

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ml_switcheroo.enums import LogicOp as LogicOp


class OpType(str, Enum):
  """Classification of the operation's syntactic usage."""

  FUNCTION = "function"
  CONTEXT = "context"
  DECORATOR = "decorator"
  ATTRIBUTE = "attribute"
  CLASS = "class"
  MACRO_GRAPH = "macro_graph"


class ContainerType(str, Enum):
  """Supported container types for argument packing."""

  TUPLE = "Tuple"
  LIST = "List"


class ParameterDef(BaseModel):
  """Definition of a single argument in an abstract operation signature.

  Updated to support semantic constraints (min, max, options), tensor properties
  (rank, dtype, shape), and function signature attributes (variadic, kind) to
  guide fuzzing and validation logic.
  """

  name: str = Field(..., description="Argument name (e.g. 'dim').")
  type: str | None = Field("Any", description="Type hint string (e.g. 'int', 'Tensor').")
  doc: str | None = Field(None, description="Argument docstring explanation.")

  # --- Feature: Rich Defaults ---
  default: JsonValue = Field(
    None,
    description="Default value. Supports primitives (int, float, bool, str) and containers (list, dict, None).",
  )

  # --- Feature: Semantic Constraints (Bounds Checking) ---
  min: float | None = Field(None, description="Minimum numeric value (inclusive).")
  max: float | None = Field(None, description="Maximum numeric value (inclusive).")
  options: list[str | int | float | bool] | None = Field(
    None, description="List of allowed discrete values (Enumeration)."
  )

  # --- Feature: Tensor Constraints (Limitation #1 & #2 Implementation) ---
  rank: int | None = Field(
    None,
    description="Required tensor rank (e.g. 4 for NCHW inputs). If None, rank is arbitrary.",
  )
  dtype: str | None = Field(
    None,
    description="Required NumPy-style dtype for tensor inputs (e.g. 'int64', 'bool', 'float32').",
  )

  # --- Feature: Symbolic Dimensions (Limitation #3 Implementation) ---
  shape_spec: str | None = Field(
    None,
    description="Symbolic shape string indicating dimension relationships (e.g. '[B, N, C]').",
  )

  # --- Feature: Signature Structure (Limitation #8 Implementation) ---
  is_variadic: bool = Field(
    False,
    description="If True, this parameter accepts variable arguments (*args). Useful for ops like `max(*tensors)` or `cat(tensors)`.",
  )
  kind: str = Field(
    "positional_or_keyword",
    description="Parameter kind: 'positional_only', 'keyword_only', 'var_positional', etc.",
  )


class Rule(BaseModel):
  """Declarative rule for conditional logic within a variant or plugin.

  Evaluated at runtime to dynamically switch APIs.
  """

  model_config = ConfigDict(populate_by_name=True, extra="allow")

  if_arg: str = Field(..., description="Name of the standard argument to check.")
  op: LogicOp = Field(LogicOp.EQ, description="Logical operator for comparison.")
  is_val: JsonValue = Field(
    None,
    description="Value to compare against. If op='is_type', this should be 'int', 'float', 'list', 'dict', 'str', or 'bool'.",
  )
  val: JsonValue = Field(None, exclude=True, description="Alias for is_val.")
  use_api: str = Field(..., description="The target API path to use if the condition matches.")

  @model_validator(mode="before")
  @classmethod
  def _sync_val_alias(
    cls, data: dict[str, JsonValue] | typing.Hashable | typing.Sized
  ) -> dict[str, JsonValue] | typing.Hashable | typing.Sized:
    """Map `val` alias to `is_val` when constructing a `Rule`.

    Args:
        data: Raw input dictionary or model instance.

    Returns:
        Normalized input dictionary.
    """
    if isinstance(data, dict) and "val" in data and "is_val" not in data:
      data = dict(data)
      data["is_val"] = data["val"]
    return data


class ImportReq(BaseModel):
  """Structured definition for a required import, supporting aliasing."""

  module: str = Field(..., description="The name of the module to import (e.g. 'numpy').")
  alias: str | None = Field(None, description="Optional alias (e.g. 'np').")


class FrameworkVariant(BaseModel):
  """Configuration for how a specific framework implements an operation.

  Defines the API endpoint, argument mappings, required imports, and versioning constraints.
  """

  api: str | None = Field(
    None,
    description="The fully qualified API path (e.g. 'torch.nn.functional.log_softmax'). If None, implies explicit lack of support unless a plugin handles it.",
  )
  op_type: OpType | None = Field(
    None, description="Variant-specific op_type override. If set, overrides the global op_type for this framework."
  )
  args: dict[str, str | float | int | None] | None = Field(
    None,
    description="Mapping from Standard Argument Name to Framework Argument Name. If value is null, argument is dropped.",
  )
  arg_values: dict[str, dict[str, JsonValue] | JsonValue] = Field(
    default_factory=lambda: {},
    description="Map of {StandardArg: {SourceValue: TargetCode}} for enums, OR {StandardArg: TargetCode} for constant injection.",
  )
  kwargs_map: dict[str, str | None] = Field(
    default_factory=lambda: {},
    description="Mapping for specific keys within a **kwargs expansion. Values can be null to drop the key.",
  )
  inject_args: dict[str, JsonValue] = Field(
    default_factory=lambda: {},
    description="Dictionary of new arguments to inject with fixed default values (supports primitives and complex types).",
  )
  inject_defaults: bool = Field(
    default=True,
    description="If False, standard argument default values are not injected as keyword arguments during normalization.",
  )

  casts: dict[str, str] | None = Field(
    None, description="Mapping of argument names to target types (e.g. {'x': 'int32'})."
  )
  requires_plugin: str | None = Field(None, description="Name of a registered plugin hook to handle translation.")

  # --- Feature: Conditional Dispatch ---
  dispatch_rules: list[Rule] = Field(
    default_factory=list,
    description="List of conditional rules to switch APIs based on argument values at runtime.",
  )

  # --- Feature: Argument Packing (Star-Args) ---
  pack_to_tuple: str | None = Field(
    None,
    description="If set (e.g. 'axes'), collects variadic positional args into a container argument.",
  )
  pack_as: ContainerType = Field(
    ContainerType.TUPLE,
    description="Container type to use for packing ('Tuple' or 'List'). Defaults to Tuple.",
  )

  # --- Feature: Tensor Layout Permutation ---
  layout_map: dict[str, str] | None = Field(
    None,
    description="Map of arguments (or 'return') to layout transformation strings (e.g., {'x': 'NCHW->NHWC'}). Engine automatically injects permutation calls to match target layout expected by this API variant.",
  )

  # --- Transformation Logic ---
  transformation_type: str | None = Field(
    None, description="Rewrite mode (e.g. 'infix', 'inline_lambda', 'strip_context')."
  )
  operator: str | None = Field(None, description="Infix operator symbol if transformation_type='infix' (e.g. '+').")

  # --- Feature: Composite Operations (Macros) ---
  macro_template: str | None = Field(
    None,
    description="Python expression template string for composite operations. Use standard argument names as placeholders (e.g. '{x} * jax.nn.sigmoid({x})').",
  )

  # --- Output Destructuring and Adaptation ---
  output_select_index: int | None = Field(
    None,
    description="Integer index to extract from a tuple return value (e.g. 0). Structured replacement for deprecated `output_adapter`.",
  )

  # --- Feature: Output Dtype Casting (Limitation 12) ---
  output_cast: str | None = Field(
    None,
    description="Optional string representing the target dtype for the output. Wraps result in .astype(...). Example: 'jnp.int64' or 'torch.long'.",
  )

  # --- Feature: Environment Configuration ---
  min_version: str | None = Field(None, description="Minimum supported version of the framework logic.")
  max_version: str | None = Field(None, description="Maximum supported version before breaking changes occurred.")

  # --- Feature: Dynamic Import Aliasing (Limitation 13) ---
  required_imports: list[str | ImportReq] = Field(
    default_factory=list,
    description="List of imports. Supports simple strings ('import cv2') or structured dicts ({'module': 'numpy', 'alias': 'np'}).",
  )

  missing_message: str | None = Field(
    None, description="Custom error message to display if this mapping fails or is missing."
  )
  replace_with_graph: list[dict[str, JsonValue]] | None = Field(
    None, description="List of nodes defining a subgraph replacement for macro_graph operations."
  )


class PluginType(str, Enum):
  """Enumeration of supported plugin structures to generate."""

  CALL = "call_transform"
  BLOCK = "block_transform"


class PluginScaffoldDef(BaseModel):
  """Metadata for generating a new Python plugin file."""

  name: str = Field(..., description="Unique hook name (e.g. 'shard_map_rewriter').")
  type: PluginType = Field(PluginType.CALL, description="Type of AST node the plugin targets.")
  doc: str = Field("Auto-generated plugin.", description="Docstring for the generated function.")
  rules: list[Rule] = Field(
    default_factory=list,
    description="List of conditional dispatch rules to compile into the plugin.",
  )

  # --- Feature: Auto-Wire ---
  auto_wire: dict[str, JsonValue] = Field(
    default_factory=lambda: {},
    description="Dictionary defining Semantic Operations to be automatically injected via the plugin itself. Matches the structure of an ODL OperationDef JSON.",
  )


class PatternDef(BaseModel):
  """Definition of a subgraph pattern for Graph Fusion."""

  name: str = Field(..., description="Name of the pattern (e.g., 'ConvBNReLU').")
  sequence: list[str] = Field(..., description="Ordered list of Operation Kinds (e.g. ['Conv2d', 'BatchNorm', 'ReLU']).")
  replace_with: str = Field(..., description="The target Operation Kind to substitute (e.g. 'FusedConvBlock').")
  description: str | None = Field(None, description="Optional description of the fusion pattern.")


class OperationDef(BaseModel):
  """Top-level definition of a new Abstract Operation.

  Contains the semantic signature, documentation, framework implementations,
  and metadata for verification, fuzzing, and documentation generation.
  """

  operation: str = Field(..., description="The PascalCase abstract name (e.g. 'LogSoftmax').")
  description: str = Field(default="", description="Docstring summary of what the operation does.")

  state_semantics: str | None = Field(
    None, description="State semantics (e.g. 'pure' or 'stateful') as per the ODL specification."
  )

  # --- Feature: Operation Type ---
  # Classifies the op usage pattern (Function call, Context Manager, etc.)
  op_type: OpType = Field(OpType.FUNCTION, description="Syntactic type: function, context, decorator, attribute, class.")

  std_args: list[str | ParameterDef] = Field(default_factory=list, description="List of standardized arguments.")
  variants: dict[str, FrameworkVariant | None] = Field(
    ...,
    description="Map of framework keys (e.g. 'torch') to their implementation details.",
  )
  scaffold_plugins: list[PluginScaffoldDef] = Field(
    default_factory=list, description="List of plugins to scaffold in the source tree."
  )

  # --- Feature: Return Type & Side Effects ---
  return_type: str | None = Field("Any", description="Type hint for return value (e.g. 'Tensor').")
  is_inplace: bool = Field(False, description="If True, op mutates the first argument.")

  # --- Feature: Verification Configuration ---
  test_rtol: float = Field(1e-3, description="Relative tolerance for numerical equivalence tests.")
  test_atol: float = Field(1e-4, description="Absolute tolerance for numerical equivalence tests.")

  # New Feature: Mode
  verification_mode: typing.Literal["approx", "exact"] = Field(
    "approx", description="Verification mode. 'approx' uses np.allclose (fuzzy). 'exact' uses np.array_equal (equality)."
  )

  skip_fuzzing: bool = Field(False, description="If True, excludes this operation from automated fuzzing.")
  nondeterministic: bool = Field(
    False,
    description="If True, output varies between runs (e.g. random ops), relaxing exact match checks.",
  )
  differentiable: bool = Field(True, description="If True, the operation supports automatic differentiation.")

  # --- Feature: Documentation & Constraints ---
  documentation_url: str | None = Field(None, description="URL to the official standard documentation for this op.")
  device_constraints: list[str] = Field(default=["cpu", "gpu"], description="List of supported hardware devices.")
  output_shape_calc: str | None = Field(None, description="Lambda string to calculate output shape from input shape.")

  # --- Feature: Cost Complexity (Limitation 19) ---
  complexity: str | None = Field(None, description="Time complexity string (e.g. 'O(N^2)').")
  sharding_supported: bool = Field(
    False, description="If True, the operation natively supports distributed sharding or can be wrapped."
  )
