"""RDNA Emitter (Backend).

Converts RDNA AST nodes into formatted assembly text.
"""

import typing

if typing.TYPE_CHECKING:
  from ml_ecosystem_snapshots.grounding.models import GroundingReport

from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter
from ml_switcheroo.core.compiler.isa.cst import IsaNode


class RdnaEmitter:
  """Convert RDNA AST nodes into textual assembly code."""

  def emit(self, nodes: typing.Sequence[IsaNode]) -> str:
    """Generate the RDNA source string from a list of nodes.

    Args:
        nodes: A sequence of IsaNode instances to format.

    Returns:
        str: The generated RDNA assembly text.
    """
    printer = RdnaPrinter()
    return printer.emit(nodes)

  def validate_instruction(
    self,
    mnemonic: str,
    gfx_arch: str = "gfx1100",
    operands: list[str] | None = None,
    modifiers: list[str] | None = None,
  ) -> "GroundingReport | None":
    """Validate an RDNA instruction mnemonic against grounded hardware specifications.

    Args:
        mnemonic: The instruction mnemonic (e.g. 'v_fma_f32').
        gfx_arch: The target graphics architecture (defaults to 'gfx1100').
        operands: Optional register/operand tokens.
        modifiers: Optional instruction modifiers.

    Returns:
        Optional[GroundingReport]: A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.hardware import validate_rdna_instruction

      report: "GroundingReport" = validate_rdna_instruction(
        mnemonic=mnemonic,
        gfx_arch=gfx_arch,
        operands=operands or ["v0", "v1", "v2"],
        modifiers=modifiers,
      )
      return report
    except (ImportError, ModuleNotFoundError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
      return None
