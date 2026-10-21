"""Type definitions for the Rewriter module."""

import pydantic


class SignatureContext(pydantic.BaseModel):
  """Track the state of the current function scope being visited.

  Used by the StructureMixin to maintain context about arguments,
  injections, and initialization status during AST traversal.
  """

  model_config = pydantic.ConfigDict(extra="forbid")

  existing_args: set[str] = pydantic.Field(default_factory=set)
  injected_args: list[tuple[str, str | None]] = pydantic.Field(default_factory=list)
  preamble_stmts: list[str] = pydantic.Field(default_factory=list)
  is_init: bool = False
  is_module_method: bool = False
