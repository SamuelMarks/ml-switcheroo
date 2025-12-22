"""
Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`k_array_api.json`, etc.)
and the Framework Configuration blocks.

Updated to support richer Argument Definitions (Dictionaries with constraints)
in `std_args` and Conditional Dispatch Rules.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict

from ml_switcheroo.enums import SemanticTier, LogicOp

# Re-exporting LogicOp just in case it's used directly,
# although import from enums is preferred.
# LogicOp = LogicOp


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
  requires_plugin: Optional[str] = Field(None, description="Name of a registered plugin hook to handle translation.")

  # --- Feature: Conditional Dispatch ---
  dispatch_rules: List[Rule] = Field(
    default_factory=list,
    description="List of conditional rules to switch APIs based on argument values at runtime.",
  )

  transformation_type: Optional[str] = Field(None, description="Special rewrite mode (e.g. 'infix', 'inline_lambda').")
  operator: Optional[str] = Field(None, description="If transformation_type='infix', the symbol to use (e.g. '+').")

  output_adapter: Optional[str] = Field(
    None,
    description=(
      "Python lambda string to normalize the return value. "
      "Wraps the target call result. Example: 'lambda x: x[0]' "
      "converts a tuple return '(val, idx)' to just 'val'."
    ),
  )


class OpDefinition(BaseModel):
  """
  Definition of an Abstract Operation standard.
  """

  model_config = ConfigDict(extra="allow")

  description: Optional[str] = None
  from_file: Optional[str] = Field(None, alias="from")

  # std_args can be simple ["x"], typed [("x", "int")], or rich Dicts [{"name": "x", "min": 0}]
  std_args: List[Union[str, Tuple[str, str], Dict[str, Any]]] = Field(
    default_factory=list, description="List of standard argument names, types, or rich definitions."
  )

  variants: Dict[str, Optional[Variant]] = Field(
    default_factory=dict, description="Map of framework keys ('torch', 'jax') to implementation details."
  )


class SemanticsFile(BaseModel):
  """Schema representing an entire semantics JSON file."""

  model_config = ConfigDict(extra="allow")

  # Meta blocks
  frameworks: Optional[Dict[str, FrameworkTraits]] = Field(None, alias="__frameworks__")
  imports: Optional[Dict[str, Any]] = Field(None, alias="__imports__")

  # Operations map (everything else)
  pass
