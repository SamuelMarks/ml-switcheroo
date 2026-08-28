"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.core.mlir.naming import NamingContext


def test_empty_while_else_elif() -> None:
  """Docstring."""
  emitter: StableHloEmitter = StableHloEmitter(NamingContext())
  code: str = """
while True:
    pass

if True:
    pass
else:
    pass

if True:
    pass
elif False:
    pass
"""
  module: cst.Module = cst.parse_module(code)
  for stmt in module.body:
    emitter._emit_statement(stmt)
