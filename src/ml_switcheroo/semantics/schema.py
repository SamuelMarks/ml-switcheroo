"""
Pydantic Schemas for Semantic Knowledge Base.

This module defines the data structure of the JSON mapping files (`k_array_api.json`, etc.).
It serves as the formal specification for:
1.  **OpDefinition**: The top-level abstract operation entry.
2.  **Variant**: Framework-specific implementation details.
3.  **Output Normalization**: The `output_adapter` field used to align return signatures.
"""

from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict


class Variant(BaseModel):
  """
  Defines how a specific framework implements an abstract operation.
  """

  model_config = ConfigDict(extra="allow")

  api: Optional[str] = Field(None, description="Fully qualified API path (e.g. 'jax.numpy.sum').")
  args: Optional[Dict[str, str]] = Field(None, description="Map of 'Standard Name' -> 'Framework Name' for arguments.")
  requires_plugin: Optional[str] = Field(None, description="Name of a registered plugin hook to handle translation.")
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

  # std_args can be simple ["x"] or typed [("x", "int")].
  # Pydantic validates structural compatibility.
  std_args: List[Union[str, Tuple[str, str]]] = Field(
    default_factory=list, description="List of standard argument names/types defined by the spec."
  )

  variants: Dict[str, Optional[Variant]] = Field(
    default_factory=dict, description="Map of framework keys ('torch', 'jax') to implementation details."
  )
