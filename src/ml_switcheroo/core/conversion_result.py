"""Data structures representing the output of the conversion pipeline.

This module defines the `ConversionResult` Pydantic model, which encapsulates
the generated code, any errors encountered, and the execution trace logs.
"""

from pydantic import BaseModel, Field, JsonValue

from ml_switcheroo.core.runtime_models import RuntimeTraceReport


class ConversionResult(BaseModel):
  """Container for the results of a transpilation job."""

  code: str = Field(default="", description="The generated source code.")
  errors: list[str] = Field(default_factory=list, description="List of error messages encountered.")
  warnings: list[str] = Field(
    default_factory=list,
    description="Diagnostic warnings and static safety recommendations.",
  )
  success: bool = Field(
    default=True,
    description="True if the pipeline completed without fatal crashes.",
  )
  trace_events: list[dict[str, JsonValue]] = Field(default_factory=list, description="Execution trace log data.")
  runtime_trace: RuntimeTraceReport | None = Field(
    default=None,
    description="On-device forward and backward pass telemetry report.",
  )

  @property
  def has_errors(self) -> bool:
    """Check if the result contains any error messages.

    Returns:
        True if one or more errors are present.

    """
    return len(self.errors) > 0

  @property
  def has_warnings(self) -> bool:
    """Check if the result contains any diagnostic warnings.

    Returns:
        True if one or more warnings are present.

    """
    return len(self.warnings) > 0
