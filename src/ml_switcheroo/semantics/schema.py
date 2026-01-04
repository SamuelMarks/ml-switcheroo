"""
Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`k_array_api.json`, etc.)
and the Framework Configuration blocks.

Updated to support:
- `PluginTraits`: Capability flags for decoupling plugins from framework identifiers.
- Rich Argument Definitions in `std_args` with Typed Defaults (Any).
- Conditional Dispatch Rules.
- **PatternDef**: Graph fusion patterns.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Set, Literal
from pydantic import BaseModel, Field, ConfigDict

from ml_switcheroo.enums import SemanticTier, LogicOp
from ml_switcheroo.core.dsl import (
  ImportReq,
  OpType,
  ContainerType,
  ParameterDef,
  Rule,
  FrameworkVariant,
  PluginType,
  PluginScaffoldDef,
  PatternDef,  # Added re-export
  OperationDef,  # Added re-export
)


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


# Re-expose Variant to maintain compatibility with manager.py usage
Variant = FrameworkVariant


class OpDefinition(OperationDef):
  """
  Compat alias for OperationDef to match existing codebase usage.
  """

  pass


class SemanticsFile(BaseModel):
  """Schema representing an entire semantics JSON file."""

  model_config = ConfigDict(extra="allow")
  frameworks: Optional[Dict[str, FrameworkTraits]] = Field(None, alias="__frameworks__")
  imports: Optional[Dict[str, Any]] = Field(None, alias="__imports__")
  patterns: Optional[List[PatternDef]] = Field(default_factory=list, alias="__patterns__")
