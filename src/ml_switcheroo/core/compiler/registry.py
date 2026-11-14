"""Compiler Registry.

Centralizes registration of Frontends (Source -> IR) and Backends (IR -> Target)
for the compiler pipeline.

This registry maps framework keys to their respective compiler components.
It cleanly separates Low-Level ISAs (NVIDIA_SASS, RDNA) which use the explicit
Graph/Compiler pipeline, from High-Level frameworks (Torch, JAX, MLIR, TikZ)
which flow through the CST Rewriter pipeline.
"""

import typing

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.compiler.ir import LogicalGraph
  from ml_switcheroo.semantics.manager import SemanticsManager

from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.backends.compiler_bridge import (
  BRIDGED_COMPILER_TARGETS,
  CompilerGraphBackendAdapter,
)
from ml_switcheroo.core.compiler.backends.html import HtmlBackend
from ml_switcheroo.core.compiler.backends.ir import IrBackend
from ml_switcheroo.core.compiler.backends.mlir_backend import MlirBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend as NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.python import PythonBackend as PythonBackend
from ml_switcheroo.core.compiler.backends.rdna import RdnaBackend
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend, TikzBackend
from ml_switcheroo.core.compiler.frontends.ir import IrFrontend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.python import PythonFrontend
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser
from ml_switcheroo.core.cst.base import CSTNode
from ml_switcheroo.core.mlir.stablehlo_parser import StableHloParser

_IsaStmtT = typing.TypeVar("_IsaStmtT", bound=CSTNode)
_IsaStmtT_co = typing.TypeVar("_IsaStmtT_co", bound=CSTNode, covariant=True)
_IsaStmtT_contra = typing.TypeVar("_IsaStmtT_contra", bound=CSTNode, contravariant=True)


class IsaParsedProgramProtocol(typing.Protocol[_IsaStmtT_co]):
  """Protocol for parsed ISA AST containers exposing `statements`."""

  @property
  def statements(self) -> typing.Sequence[_IsaStmtT_co]:
    """Return parsed ISA statement sequence.

    Returns:
        Sequence of parsed ISA statement nodes.
    """


class _IsaParserInstanceProtocol(typing.Protocol[_IsaStmtT_co]):
  """Protocol for instantiated ISA parsers."""

  def parse(self) -> IsaParsedProgramProtocol[_IsaStmtT_co]:
    """Parse raw assembly text into a program container.

    Returns:
        Parsed program containing `statements`.
    """


class IsaParserClassProtocol(typing.Protocol[_IsaStmtT_co]):
  """Protocol for ISA parser classes registered as `(ParserCls, LifterCls)` in `_FRONTENDS`."""

  def __call__(self, code: str) -> _IsaParserInstanceProtocol[_IsaStmtT_co]:
    """Instantiate parser with raw assembly source text.

    Args:
        code: Raw ISA assembly text.

    Returns:
        Initialized ISA parser instance.
    """


class _IsaLifterInstanceProtocol(typing.Protocol[_IsaStmtT_contra]):
  """Protocol for instantiated ISA lifters."""

  def lift(self, nodes: typing.Sequence[_IsaStmtT_contra]) -> "LogicalGraph":
    """Lift parsed ISA statements into a `LogicalGraph`.

    Args:
        nodes: Sequence of parsed ISA statement nodes.

    Returns:
        Lifted `LogicalGraph`.
    """


class IsaLifterClassProtocol(typing.Protocol[_IsaStmtT_contra]):
  """Protocol for ISA lifter classes registered as `(ParserCls, LifterCls)` in `_FRONTENDS`."""

  def __call__(self) -> _IsaLifterInstanceProtocol[_IsaStmtT_contra]:
    """Instantiate lifter.

    Returns:
        Initialized ISA lifter instance.
    """


@typing.runtime_checkable
class IsaFrontendProtocol(typing.Protocol):
  """Protocol for registered `(ParserCls, LifterCls)` ISA frontend entries."""

  def lift_code(self, code: str) -> "LogicalGraph":
    """Parse and lift raw ISA source text into a `LogicalGraph`.

    Args:
        code: Raw ISA assembly text.

    Returns:
        Lifted `LogicalGraph`.
    """


class IsaFrontendEntry(typing.Generic[_IsaStmtT]):
  """Typed `(ParserCls, LifterCls)` pair registered in `_FRONTENDS`.

  Attributes:
      parser_cls: ISA parser constructor.
      lifter_cls: ISA lifter constructor.
  """

  def __init__(
    self,
    parser_cls: IsaParserClassProtocol[_IsaStmtT],
    lifter_cls: IsaLifterClassProtocol[_IsaStmtT],
  ) -> None:
    """Bind an ISA parser class and lifter class together.

    Args:
        parser_cls: ISA parser constructor.
        lifter_cls: ISA lifter constructor.
    """
    self.parser_cls = parser_cls
    self.lifter_cls = lifter_cls

  def lift_code(self, code: str) -> "LogicalGraph":
    """Parse ``code`` with ``parser_cls`` and lift statements with ``lifter_cls``.

    Args:
        code: Raw ISA source code string.

    Returns:
        Lifted `LogicalGraph` instance.
    """
    return self.lifter_cls().lift(self.parser_cls(code).parse().statements)


FrontendSpec: typing.TypeAlias = (
  type[PythonFrontend] | type[IrFrontend] | type[StableHloParser] | type["GraphFrontend"] | IsaFrontendProtocol
)


class BaseFrontend:
  """Abstract base class for compiler frontends in the registry.

  This class serves as a marker/typing base for all source-to-IR frontends.
  """


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

    Raises:
        NotImplementedError: Must be implemented by concrete frontend subclasses.
    """
    raise NotImplementedError


# Backend mappings for the Compiler (Graph -> Text) pipeline
_BACKENDS: dict[str, type[CompilerBackend]] = {
  "nvidia_sass": NvidiaSassBackend,
  "rdna": RdnaBackend,
  "python": PythonBackend,
  # Extras
  "html": HtmlBackend,
  "tikz": TikzBackend,
  "latex_dsl": LatexBackend,
  "mlir": MlirBackend,
  "stablehlo": StableHloBackend,
  "ir": IrBackend,
  "ml_switcheroo_ir": IrBackend,
  **{tgt: CompilerGraphBackendAdapter for tgt in BRIDGED_COMPILER_TARGETS},
}

# Frontend mappings for the Compiler (Text -> Graph) pipeline
_FRONTENDS: dict[str, FrontendSpec] = {
  "python": PythonFrontend,
  "ir": IrFrontend,
  "ml_switcheroo_ir": IrFrontend,
  "nvidia_sass": IsaFrontendEntry(NvidiaSassParser, NvidiaSassLifter),
  "rdna": IsaFrontendEntry(RdnaParser, RdnaLifter),
  "stablehlo": StableHloParser,
}


def lift_isa_pair(
  pair: tuple[IsaParserClassProtocol[_IsaStmtT], IsaLifterClassProtocol[_IsaStmtT]],
  code: str,
) -> "LogicalGraph":
  """Parse and lift ISA source code using a `(ParserCls, LifterCls)` pair.

  Args:
      pair: `(ParserCls, LifterCls)` tuple.
      code: Raw ISA source code string.

  Returns:
      Lifted `LogicalGraph` instance.
  """
  parser_cls, lifter_cls = pair
  return IsaFrontendEntry(parser_cls, lifter_cls).lift_code(code)


def get_backend_class(target: str) -> type[CompilerBackend] | None:
  """Return the backend class for the target (e.g. 'nvidia_sass').

  Args:
      target: The target framework identifier.

  Returns:
      The backend class type or PythonBackend if not found, or None.

  """
  return _BACKENDS.get(target, _BACKENDS.get("python"))


def instantiate_backend(
  backend_cls: type[CompilerBackend],
  target: str,
  semantics: "SemanticsManager | None" = None,
) -> CompilerBackend:
  """Instantiate a compiler backend class for the given target framework.

  Args:
      backend_cls: The resolved ``CompilerBackend`` subclass to instantiate.
      target: The target framework or dialect identifier.
      semantics: Optional semantics manager for operator resolution.

  Returns:
      Initialized ``CompilerBackend`` instance.
  """
  if backend_cls.__name__ == "PythonBackend":
    return backend_cls(framework=target, semantics=semantics)
  if backend_cls.__name__ == "IrBackend":
    ir_format = "python" if target == "ml_switcheroo_ir" else "json"
    return backend_cls(semantics=semantics, format=ir_format)
  if backend_cls.__name__ == "CompilerGraphBackendAdapter":
    return CompilerGraphBackendAdapter(semantics=semantics, target=target)
  return backend_cls(semantics)


def is_isa_target(target: str) -> bool:
  """Determine if the target requires the Graph Compiler pipeline.

  Only Low-Level Assembly targets handling Registers or Visualization
  backends that strictly consume Graphs are routed here.


  Note: MLIR/TikZ/Latex/HTML/RDNA/NVIDIA_SASS/IR use this path for graph-based generation
  if selected as target in CLI, bypassing the CST rewriter.

  Args:
      target: The target framework identifier.

  Returns:
      True if the target is an ISA or Graph-based format.

  """
  backend_cls = _BACKENDS.get(target)
  return backend_cls is not None and backend_cls not in (PythonBackend, StableHloBackend)


def is_isa_source(source: str) -> bool:
  """Determine if the source requires Lifting (ASM/IR -> Graph -> AST).

  NVIDIA_SASS, RDNA, StableHLO, and IR are treated as low-level or graph source inputs.

  Args:
      source: The source framework identifier.

  Returns:
      True if the source is an ISA or graph source requiring lifting.

  """
  frontend_spec = _FRONTENDS.get(source)
  return frontend_spec is not None and frontend_spec is not PythonFrontend
