"""Compiler Registry.

Centralizes registration of Frontends (Source -> IR) and Backends (IR -> Target)
for the compiler pipeline.

This registry maps framework keys to their respective compiler components.
It cleanly separates Low-Level ISAs (NVIDIA_SASS, RDNA) which use the explicit
Graph/Compiler pipeline, from High-Level frameworks (Torch, JAX, MLIR, TikZ)
which flow through the CST Rewriter pipeline.
"""

from typing import Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
  from ml_switcheroo.core.compiler.ir import LogicalGraph

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassParser, NvidiaSassLifter
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.rdna import RdnaParser, RdnaLifter
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.backends.mlir_backend import MlirBackend
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend, LatexBackend
from ml_switcheroo.core.compiler.backends.ir import IrBackend
from ml_switcheroo.core.compiler.frontends.ir import IrFrontend
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser


class BaseFrontend:
  """Abstract base class for compiler frontends in the registry.

  This class serves as a marker/typing base for all source-to-IR frontends.
  """

  pass


class GraphFrontend(BaseFrontend):
  """Produce LogicalGraph from code via parse/lift chain.

  This frontend takes source code strings and converts them into
  logical graph representations used by compiler backends.
  """

  def parse_to_graph(self, code: str) -> "LogicalGraph":
    """Parse source code into a logical graph representation.

    Args:
        code: The source code string to parse.

    Returns:
        The generated logical graph or equivalent representation.
    """
    raise NotImplementedError


# Backend mappings for the Compiler (Graph -> Text) pipeline
_BACKENDS: Dict[str, Type[CompilerBackend]] = {
  "nvidia_sass": NvidiaSassBackend,
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
  "ir": IrBackend,
  "ml_switcheroo_ir": IrBackend,
}

# Frontend mappings for the Compiler (Text -> Graph) pipeline
_FRONTENDS = {
  "python": PythonFrontend,
  "torch": PythonFrontend,
  "jax": PythonFrontend,
  "flax_nnx": PythonFrontend,
  "keras": PythonFrontend,
  "tensorflow": PythonFrontend,
  "numpy": PythonFrontend,
  "mlx": PythonFrontend,
  "paxml": PythonFrontend,
  "ir": IrFrontend,
  "ml_switcheroo_ir": IrFrontend,
  # ISAs use Parser+Lifter tuple strategy handled by engine
  "nvidia_sass": (NvidiaSassParser, NvidiaSassLifter),
  "rdna": (RdnaParser, RdnaLifter),
  "stablehlo": StableHloParser,
}


def get_backend_class(target: str) -> Optional[Type[CompilerBackend]]:
  """Return the backend class for the target (e.g. 'nvidia_sass').

  Args:
      target: The target framework identifier.

  Returns:
      The backend class type or PythonBackend if not found, or None.

  """
  return _BACKENDS.get(target, _BACKENDS.get("python"))


def is_isa_target(target: str) -> bool:
  """Determine if the target requires the Graph Compiler pipeline.

  Only Low-Level Assembly targets handling Registers or Visualization
  backends that strictly consume Graphs are routed here.


  Note: MLIR/StableHLO/TikZ/Latex/HTML/RDNA/NVIDIA_SASS/IR use this path for graph-based generation
  if selected as target in CLI, bypassing the CST rewriter.

  Args:
      target: The target framework identifier.

  Returns:
      True if the target is an ISA or Graph-based format.

  """
  return target in ["nvidia_sass", "rdna", "html", "tikz", "latex_dsl", "mlir", "ir", "ml_switcheroo_ir"]


def is_isa_source(source: str) -> bool:
  """Determine if the source requires Lifting (ASM/IR -> Graph -> AST).

  NVIDIA_SASS, RDNA, StableHLO, and IR are treated as low-level or graph source inputs.

  Args:
      source: The source framework identifier.

  Returns:
      True if the source is an ISA or graph source requiring lifting.

  """
  return source in ["nvidia_sass", "rdna", "stablehlo", "ir", "ml_switcheroo_ir"]
