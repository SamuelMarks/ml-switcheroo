"""
Compiler Registry.

Centralizes registration of Frontends (Source -> IR) and Backends (IR -> Target)
for the compiler pipeline.

This registry maps framework keys to their respective compiler components.
It cleanly separates Low-Level ISAs (SASS, RDNA) which use the explicit
Graph/Compiler pipeline, from High-Level frameworks (Torch, JAX, MLIR, TikZ)
which flow through the CST Rewriter pipeline.
"""

from typing import Dict, Optional, Type, Any

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.frontends.sass import SassParser, SassLifter
from ml_switcheroo.compiler.backends.sass import SassBackend
from ml_switcheroo.compiler.frontends.rdna import RdnaParser, RdnaLifter
from ml_switcheroo.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.compiler.backends.python import PythonBackend
from ml_switcheroo.compiler.frontends.python import PythonFrontend
from ml_switcheroo.compiler.backends.extras import (
  HtmlBackend,
  TikzBackend,
  LatexBackend,
  MlirBackend,
  StableHloBackend,
)


class BaseFrontend:
  """Abstract base for registry typing."""

  pass


class GraphFrontend(BaseFrontend):
  """Produces LogicalGraph from code via parse/lift chain."""

  def parse_to_graph(self, code: str) -> Any: ...


# Backend mappings for the Compiler (Graph -> Text) pipeline
_BACKENDS: Dict[str, Type[CompilerBackend]] = {
  "sass": SassBackend,
  "rdna": RdnaBackend,
  "python": PythonBackend,
  # High-Level Fallbacks (if routed to compiler)
  "torch": PythonBackend,
  "jax": PythonBackend,
  "flax_nnx": PythonBackend,
  "keras": PythonBackend,
  "tensorflow": PythonBackend,
  "numpy": PythonBackend,
  "mlx": PythonBackend,
  "paxml": PythonBackend,
  # Extras
  "html": HtmlBackend,
  "tikz": TikzBackend,
  "latex_dsl": LatexBackend,
  "mlir": MlirBackend,
  "stablehlo": StableHloBackend,
}

# Frontend mappings for the Compiler (Text -> Graph) pipeline
_FRONTENDS: Dict[str, Any] = {
  "python": PythonFrontend,
  "torch": PythonFrontend,
  "jax": PythonFrontend,
  "flax_nnx": PythonFrontend,
  "keras": PythonFrontend,
  "tensorflow": PythonFrontend,
  "numpy": PythonFrontend,
  "mlx": PythonFrontend,
  "paxml": PythonFrontend,
  # ISAs use Parser+Lifter tuple strategy handled by engine
  "sass": (SassParser, SassLifter),
  "rdna": (RdnaParser, RdnaLifter),
}


def get_backend_class(target: str) -> Optional[Type[CompilerBackend]]:
  """Returns the backend class for the target (e.g. 'sass')."""
  return _BACKENDS.get(target, _BACKENDS.get("python"))


def is_isa_target(target: str) -> bool:
  """
  Determines if the target requires the Graph Compiler pipeline.

  Only Low-Level Assembly targets handling Registers or Visualization
  backends that strictly consume Graphs are routed here.

  Note: MLIR/StableHLO/TikZ/Latex are high-level structural representations,
  so they are routed through the Rewriter/CST pipeline (via Emitters)
  for maximum fidelity, unless explicitly requested via CLI intermediate flags.
  """
  return target in ["sass", "rdna", "html"]


def is_isa_source(source: str) -> bool:
  """
  Determines if the source requires Lifting (ASM -> Graph -> AST).
  Only SASS and RDNA are treated as low-level source inputs.
  """
  return source in ["sass", "rdna"]
