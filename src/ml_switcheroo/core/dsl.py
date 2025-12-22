"""
Operation Definition Language (ODL) Schema.

This module defines the Pydantic models used to validate the input (e.g. YAML)
for defining new operations. It represents the source of truth for generating
new operation definitions in the codebase, enforcing schema validity before
AST injection occurs.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class ParameterDef(BaseModel):
  """
  Definition of a single argument in an abstract operation signature.
  """

  name: str = Field(..., description="Argument name (e.g. 'dim').")
  type: Optional[str] = Field("Any", description="Type hint string (e.g. 'int', 'Tensor').")
  doc: Optional[str] = Field(None, description="Argument docstring explanation.")
  default: Optional[str] = Field(None, description="Default value as a string code representation.")


class FrameworkVariant(BaseModel):
  """
  Configuration for how a specific framework implements the operation.
  """

  api: str = Field(..., description="The fully qualified API path (e.g. 'torch.nn.functional.relu').")
  args: Optional[Dict[str, str]] = Field(
    None, description="Mapping from Standard Argument Name to Framework Argument Name."
  )
  casts: Optional[Dict[str, str]] = Field(
    None, description="Mapping of argument names to target types (e.g. {'x': 'int32'})."
  )
  requires_plugin: Optional[str] = Field(None, description="Name of a plugin hook required to handle this operation.")
  transformation_type: Optional[str] = Field(None, description="Rewrite mode (e.g. 'infix', 'inline_lambda').")
  output_adapter: Optional[str] = Field(None, description="Lambda string to normalize return values.")


class PluginType(str, Enum):
  """
  Enumeration of supported plugin structures to generate.
  """

  CALL = "call_transform"
  BLOCK = "block_transform"


class Rule(BaseModel):
  """
  Declarative rule for conditional logic within a plugin.
  Used to generate switch-statements in the scaffolded code.
  """

  model_config = ConfigDict(populate_by_name=True)

  if_arg: str = Field(..., description="Name of the keyword argument to check in the source call.")
  is_val: Union[str, int, float, bool] = Field(
    ..., alias="is", description="Literal value to match (eq check). Matches Python types."
  )
  use_api: str = Field(..., description="The target API path to use if the condition matches.")


class PluginScaffoldDef(BaseModel):
  """
  Metadata for generating a new Python plugin file.
  """

  name: str = Field(..., description="Unique hook name (e.g. 'shard_map_rewriter').")
  type: PluginType = Field(PluginType.CALL, description="Type of AST node the plugin targets.")
  doc: str = Field("Auto-generated plugin.", description="Docstring for the generated function.")
  rules: List[Rule] = Field(
    default_factory=list, description="List of conditional dispatch rules to compile into the plugin."
  )


class OperationDef(BaseModel):
  """
  Top-level definition of a new Abstract Operation.
  """

  operation: str = Field(..., description="The PascalCase abstract name (e.g. 'LogSoftmax').")
  description: str = Field(..., description="Docstring summary of what the operation does.")
  std_args: List[ParameterDef] = Field(default_factory=list, description="List of standardized arguments.")
  variants: Dict[str, FrameworkVariant] = Field(
    ..., description="Map of framework keys (e.g. 'torch') to their implementation details."
  )
  scaffold_plugins: List[PluginScaffoldDef] = Field(
    default_factory=list, description="List of plugins to scaffold in the source tree."
  )
