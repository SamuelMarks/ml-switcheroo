"""Standardized Assembly Printer for Hardware ISAs."""

import typing

from ml_switcheroo.core.compiler.isa.cst import IsaComment, IsaInstruction, IsaLabel, IsaNode


class IsaPrinter:
  """Standardized Assembly Printer with configurable formatting."""

  def __init__(self, indent: int = 4, operand_tab_stop: int = 16, upper_case_opcode: bool = False) -> None:
    """Initialize the ISA assembly printer.

    Args:
        indent: Number of leading spaces for instructions (default: 4).
        operand_tab_stop: Column tab stop for aligning instruction operands (default: 16).
        upper_case_opcode: Whether to format instruction mnemonics in uppercase (default: False).
    """
    self.indent = indent
    self.operand_tab_stop = operand_tab_stop
    self.upper_case_opcode = upper_case_opcode
    self._current_indent = 0

  def emit(self, nodes: typing.Sequence[IsaNode]) -> str:
    """Render a sequence of ISA CST nodes into formatted assembly text.

    Args:
        nodes: Sequence of IsaNode instances to render.

    Returns:
        Formatted assembly text string ending with a newline.
    """
    lines = []
    for node in nodes:
      if isinstance(node, IsaLabel):
        # Labels usually unindented
        lines.append(node.to_text())
      elif isinstance(node, IsaInstruction):
        opcode = node.opcode.upper() if self.upper_case_opcode else node.opcode.lower()
        prefix = " " * self.indent
        if node.operands:
          operands_str = ", ".join(op.to_text().strip() for op in node.operands)
          # Support predicate via duck typing if present
          pred = getattr(node, "predicate", "")
          if pred:
            opcode_field = f"@{pred} {opcode}"
          else:
            opcode_field = opcode

          padding = max(1, self.operand_tab_stop - len(opcode_field))
          lines.append(f"{prefix}{opcode_field}{' ' * padding}{operands_str};")
        else:
          lines.append(f"{prefix}{opcode};")
      elif isinstance(node, IsaComment):
        lines.append(node.to_text())
      else:
        lines.append(node.to_text())
    return "\n".join(lines) + "\n"
