"""
Operation Definition Language (ODL) Schema.

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

from typing import Dict, List, Optional, Any, Union, Tuple, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from ml_switcheroo.enums import LogicOp


class OpType(str, Enum):
  """
  Classification of the operation's syntactic usage.
  """

  FUNCTION = "function"
  CONTEXT = "context"
  DECORATOR = "decorator"


class ContainerType(str, Enum):
  """
  Supported container types for argument packing.
  """

  TUPLE = "Tuple"
  LIST = "List"


class ParameterDef(BaseModel):
  """
  Definition of a single argument in an abstract operation signature.

  Updated to support semantic constraints (min, max, options), tensor properties
  (rank, dtype, shape), and function signature attributes (variadic, kind) to
  guide fuzzing and validation logic.
  """

  name: str = Field(..., description="Argument name (e.g. 'dim').")
  type: Optional[str] = Field("Any", description="Type hint string (e.g. 'int', 'Tensor').")
  doc: Optional[str] = Field(None, description="Argument docstring explanation.")

  # --- Feature: Rich Defaults ---
  default: Optional[Any] = Field(
    None, description="Default value. Supports primitives (int, float, bool, str) and containers (list, dict, None)."
  )

  # --- Feature: Semantic Constraints (Bounds Checking) ---
  min: Optional[float] = Field(None, description="Minimum numeric value (inclusive).")
  max: Optional[float] = Field(None, description="Maximum numeric value (inclusive).")
  options: Optional[List[Union[str, int, float, bool]]] = Field(
    None, description="List of allowed discrete values (Enumeration)."
  )

  # --- Feature: Tensor Constraints (Limitation #1 & #2 Implementation) ---
  rank: Optional[int] = Field(
    None,
    description="Required tensor rank (e.g. 4 for NCHW inputs). If None, rank is arbitrary.",
  )
  dtype: Optional[str] = Field(
    None,
    description="Required numpy-style dtype for tensor inputs (e.g. 'int64', 'bool', 'float32').",
  )

  # --- Feature: Symbolic Dimensions (Limitation #3 Implementation) ---
  shape_spec: Optional[str] = Field(
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
  """
  Declarative rule for conditional logic within a variant or plugin.
  Evaluated at runtime to dynamically switch APIs.
  """

  model_config = ConfigDict(populate_by_name=True)

  if_arg: str = Field(..., description="Name of the standard argument to check.")
  op: LogicOp = Field(LogicOp.EQ, description="Logical operator for comparison.")
  is_val: Any = Field(
    ...,
    alias="val",
    description="Value to compare against. If op='is_type', this should be 'int', 'float', 'list', 'dict', 'str', or 'bool'.",
  )
  use_api: str = Field(..., description="The target API path to use if the condition matches.")


class ImportReq(BaseModel):
  """
  Structured definition for a required import, supporting aliasing.
  """

  module: str = Field(..., description="The name of the module to import (e.g. 'numpy').")
  alias: Optional[str] = Field(None, description="Optional alias (e.g. 'np').")


class FrameworkVariant(BaseModel):
  """
  Configuration for how a specific framework implements an operation.

  Defines the API endpoint, argument mappings, required imports, and versioning constraints.
  """

  api: Optional[str] = Field(
    None,
    description="The fully qualified API path (e.g. 'torch.nn.functional.log_softmax'). If None, implies explicit lack of support unless a plugin handles it.",
  )
  args: Optional[Dict[str, str]] = Field(
    None, description="Mapping from Standard Argument Name to Framework Argument Name."
  )
  arg_values: Optional[Dict[str, Dict[str, str]]] = Field(
    None,
    description="Map of {StandardArg: {SourceValueString: TargetValueCode}} for enum value mapping. Keys should match stringified source values (e.g. 'mean' or '0'). Values are target code strings.",
  )
  kwargs_map: Optional[Dict[str, str]] = Field(None, description="Mapping for specific keys within a **kwargs expansion.")

  inject_args: Optional[Dict[str, Any]] = Field(
    None,
    description="Dictionary of new arguments to inject with fixed default values (supports primitives and complex types).",
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

  # --- Feature: Argument Packing (Star-Args) ---
  pack_to_tuple: Optional[str] = Field(
    None,
    description="If set (e.g. 'axes'), collects variadic positional args into a container argument.",
  )
  pack_as: ContainerType = Field(
    ContainerType.TUPLE,
    description="Container type to use for packing ('Tuple' or 'List'). Defaults to Tuple.",
  )

  # --- Feature: Tensor Layout Permutation ---
  layout_map: Optional[Dict[str, str]] = Field(
    None,
    description="Map of arguments (or 'return') to layout transformation strings (e.g., {'x': 'NCHW->NHWC'}). "
    "Engine automatically injects permutation calls to match target layout expected by this API variant.",
  )

  # --- Transformation Logic ---
  transformation_type: Optional[str] = Field(
    None, description="Rewrite mode (e.g. 'infix', 'inline_lambda', 'strip_context')."
  )
  operator: Optional[str] = Field(None, description="Infix operator symbol if transformation_type='infix' (e.g. '+').")

  # --- Feature: Composite Operations (Macros) ---
  macro_template: Optional[str] = Field(
    None,
    description="Python expression template string for composite operations. "
    "Use standard argument names as placeholders (e.g. '{x} * jax.nn.sigmoid({x})').",
  )

  # --- Output Destructuring and Adaptation ---
  output_select_index: Optional[int] = Field(
    None,
    description="Integer index to extract from a tuple return value (e.g. 0). Structured replacement for `output_adapter`.",
  )
  output_adapter: Optional[str] = Field(
    None,
    description="Lambda string to normalize return values (e.g. 'lambda x: x[0]'). Deprecated in favor of `output_select_index` for simple indexing.",
  )

  # --- Feature: Output Dtype Casting (Limitation 12) ---
  output_cast: Optional[str] = Field(
    None,
    description="Optional string representing the target dtype for the output. Wraps result in .astype(...). Example: 'jnp.int64' or 'torch.long'.",
  )

  # --- Feature: Environment Configuration ---
  min_version: Optional[str] = Field(None, description="Minimum supported version of the framework logic.")
  max_version: Optional[str] = Field(None, description="Maximum supported version before breaking changes occurred.")

  # --- Feature: Dynamic Import Aliasing (Limitation 13) ---
  required_imports: List[Union[str, ImportReq]] = Field(
    default_factory=list,
    description="List of imports. Supports simple strings ('import cv2') or structured dicts ({'module': 'numpy', 'alias': 'np'}).",
  )

  missing_message: Optional[str] = Field(
    None, description="Custom error message to display if this mapping fails or is missing."
  )


class PluginType(str, Enum):
  """
  Enumeration of supported plugin structures to generate.
  """

  CALL = "call_transform"
  BLOCK = "block_transform"


class PluginScaffoldDef(BaseModel):
  """
  Metadata for generating a new Python plugin file.
  """

  name: str = Field(..., description="Unique hook name (e.g. 'shard_map_rewriter').")
  type: PluginType = Field(PluginType.CALL, description="Type of AST node the plugin targets.")
  doc: str = Field("Auto-generated plugin.", description="Docstring for the generated function.")
  rules: List[Rule] = Field(
    default_factory=list,
    description="List of conditional dispatch rules to compile into the plugin.",
  )

  # --- Feature: Auto-Wire ---
  auto_wire: Optional[Dict[str, Any]] = Field(
    None,
    description="Dictionary defining Semantic Operations to be automatically injected via the plugin itself."
    "Matches the structure of an ODL OperationDef JSON.",
  )


class PatternDef(BaseModel):
  """
  Definition of a subgraph pattern for Graph Fusion.
  """

  name: str = Field(..., description="Name of the pattern (e.g., 'ConvBNReLU').")
  sequence: List[str] = Field(..., description="Ordered list of Operation Kinds (e.g. ['Conv2d', 'BatchNorm', 'ReLU']).")
  replace_with: str = Field(..., description="The target Operation Kind to substitute (e.g. 'FusedConvBlock').")
  allow_partial: bool = Field(False, description="If True, matches prefixes of sequence (not implemented yet).")


class OperationDef(BaseModel):
  """
  Top-level definition of a new Abstract Operation.

  Contains the semantic signature, documentation, framework implementations,
  and metadata for verification, fuzzing, and documentation generation.
  """

  operation: str = Field(..., description="The PascalCase abstract name (e.g. 'LogSoftmax').")
  description: str = Field(..., description="Docstring summary of what the operation does.")

  # --- Feature: Operation Type ---
  # Classifies the op usage pattern (Function call, Context Manager, etc.)
  op_type: OpType = Field(OpType.FUNCTION, description="Syntactic type: function, context, decorator.")

  std_args: List[Union[str, Tuple[str, str], ParameterDef, Dict[str, Any]]] = Field(
    default_factory=list, description="List of standardized arguments."
  )
  variants: Dict[str, Optional[FrameworkVariant]] = Field(
    ...,
    description="Map of framework keys (e.g. 'torch') to their implementation details.",
  )
  scaffold_plugins: List[PluginScaffoldDef] = Field(
    default_factory=list, description="List of plugins to scaffold in the source tree."
  )

  # --- Feature: Return Type & Side Effects ---
  return_type: Optional[str] = Field("Any", description="Type hint for return value (e.g. 'Tensor').")
  is_inplace: bool = Field(False, description="If True, op mutates the first argument.")

  # --- Feature: Verification Configuration ---
  test_rtol: float = Field(1e-3, description="Relative tolerance for numerical equivalence tests.")
  test_atol: float = Field(1e-4, description="Absolute tolerance for numerical equivalence tests.")

  # New Feature: Mode
  verification_mode: Literal["approx", "exact"] = Field(
    "approx", description="Verification mode. 'approx' uses np.allclose (fuzzy). 'exact' uses np.array_equal (equality)."
  )

  skip_fuzzing: bool = Field(False, description="If True, excludes this operation from automated fuzzing.")
  nondeterministic: bool = Field(
    False,
    description="If True, output varies between runs (e.g. random ops), relaxing exact match checks.",
  )
  differentiable: bool = Field(True, description="If True, the operation supports automatic differentiation.")

  # --- Feature: Documentation & Constraints ---
  deprecated: bool = Field(False, description="Marks the abstract operation as deprecated.")
  replaced_by: Optional[str] = Field(None, description="Name of the replacement operation if deprecated.")
  documentation_url: Optional[str] = Field(None, description="URL to the official standard documentation for this op.")
  device_constraints: List[str] = Field(default=["cpu", "gpu"], description="List of supported hardware devices.")
  output_shape_calc: Optional[str] = Field(None, description="Lambda string to calculate output shape from input shape.")

  # --- Feature: Cost Complexity (Limitation 19) ---
  complexity: Optional[str] = Field(None, description="Time complexity string (e.g. 'O(N^2)').")
