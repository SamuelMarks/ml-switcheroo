"""
Data structures representing the output of the conversion pipeline.

This module defines the `ConversionResult` Pydantic model, which encapsulates
the generated code, any errors encountered, and the execution trace logs.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ConversionResult(BaseModel):
  """
  Container for the results of a transpilation job.
  """

  code: str = Field(default="", description="The generated source code.")
  errors: List[str] = Field(default_factory=list, description="List of error messages encountered.")
  success: bool = Field(
    default=True,
    description="True if the pipeline completed without fatal crashes.",
  )
  trace_events: List[Dict[str, Any]] = Field(default_factory=list, description="Execution trace log data.")

  @property
  def has_errors(self) -> bool:
    """
    Check if the result contains any error messages.

    Returns:
        True if one or more errors are present.
    """
    return len(self.errors) > 0
