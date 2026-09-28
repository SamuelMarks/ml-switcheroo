"""RDNA Emitter (Backend).

Converts RDNA AST nodes into formatted assembly text.
"""

from typing import Any, List, Optional
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode
from ml_switcheroo.core.compiler.backends.rdna.printer import RdnaPrinter


class RdnaEmitter:
  """Convert RDNA AST nodes into textual assembly code."""

  def emit(self, nodes: List[RdnaNode]) -> str:
    """Generate the RDNA source string from a list of nodes.

    Args:
        nodes: A list of RdnaNode instances to format.

    Returns:
        str: The generated RDNA assembly text.
    """
    printer = RdnaPrinter()
    return printer.emit(nodes)

  def validate_instruction(
    self,
    mnemonic: str,
    gfx_arch: str = "gfx1100",
    operands: Optional[List[str]] = None,
    modifiers: Optional[List[str]] = None,
  ) -> Any:
    """Validate an RDNA instruction mnemonic against grounded hardware specifications.

    Args:
        mnemonic: The instruction mnemonic (e.g. 'v_fma_f32').
        gfx_arch: The target graphics architecture (defaults to 'gfx1100').
        operands: Optional register/operand tokens.
        modifiers: Optional instruction modifiers.

    Returns:
        Any: A GroundingReport if ml_ecosystem_snapshots is available, or None.
    """
    try:
      from ml_ecosystem_snapshots.grounding.hardware import validate_rdna_instruction

      return validate_rdna_instruction(
        mnemonic=mnemonic,
        gfx_arch=gfx_arch,
        operands=operands or ["v0", "v1", "v2"],
        modifiers=modifiers,
      )
    except Exception:
      return None
