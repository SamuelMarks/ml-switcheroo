"""AMD RDNA Assembly Printer."""

from ml_switcheroo.core.compiler.isa.printer import IsaPrinter


class RdnaPrinter(IsaPrinter):
  """Assembly printer specialized for AMD RDNA instructions with lowercase opcodes."""

  def __init__(self, indent: int = 4, operand_tab_stop: int = 16, upper_case_opcode: bool = False) -> None:
    """Initialize the RDNA assembly printer.

    Args:
        indent: Number of leading spaces for instructions (default: 4).
        operand_tab_stop: Column tab stop for aligning instruction operands (default: 16).
        upper_case_opcode: Ignored; RDNA mnemonics are always emitted in lowercase.
    """
    super().__init__(indent, operand_tab_stop, False)
