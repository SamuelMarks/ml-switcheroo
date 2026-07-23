"""Parser for the RDNA frontend."""

from typing import Tuple, List
import string

from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  LabelRef,
  Modifier,
  Operand,
  RdnaNode,
  SGPR,
  VGPR,
  Memory,
)


class RdnaParser:
  """Custom LLVM-MC style parser for AMD RDNA / GCN assembly."""

  def __init__(self, code: str) -> None:
    """Initialize the parser.

    Args:
        code (str): The raw RDNA source string.
    """
    self.code = code
    self.pos = 0
    self.length = len(code)

  def parse(self) -> List[RdnaNode]:
    """Parses the entire code block.

    Returns:
        List[RdnaNode]: A list of AST nodes.
    """
    nodes: List[RdnaNode] = []

    if not self.code.strip():
      return []

    while self.pos < self.length:
      leading_trivia = self._consume_whitespace()

      if self.pos >= self.length:
        if leading_trivia and nodes:
          nodes[-1].trailing_trivia += leading_trivia
        break

      ch = self.code[self.pos]

      if ch == ";":
        # Comment in RDNA
        self._consume(1)
        text = self._read_until("\n").strip()
        comment_node = Comment(text=text)
        comment_node.leading_trivia = leading_trivia
        comment_node.trailing_trivia = self._consume_whitespace(newlines_only=True)
        nodes.append(comment_node)
        continue

      if ch == ".":
        # Directive
        self._consume(1)
        name = self._read_identifier()
        self._consume_whitespace_inline()
        params = []
        while self.pos < self.length and self.code[self.pos] not in ("\n", ";"):
          param = self._read_until_chars((",", "\n", ";")).strip()
          if param:
            params.append(param)
          if self.pos < self.length and self.code[self.pos] == ",":
            self._consume(1)

        dir_node = Directive(name=name, params=params)
        dir_node.leading_trivia = leading_trivia
        dir_node.trailing_trivia = self._consume_whitespace(newlines_only=True)
        if self.pos < self.length and self.code[self.pos] == "\n":
          dir_node.trailing_trivia += "\n"
          self._consume(1)
        nodes.append(dir_node)
        continue

      ident = self._read_identifier(allow_dot=True)

      if self.pos < self.length and self.code[self.pos] == ":":
        # Label
        self._consume(1)
        lbl_node = Label(name=ident)
        lbl_node.leading_trivia = leading_trivia
        lbl_node.trailing_trivia = self._consume_whitespace(newlines_only=True)
        if self.pos < self.length and self.code[self.pos] == "\n":
          lbl_node.trailing_trivia += "\n"
          self._consume(1)
        nodes.append(lbl_node)
        continue

      # Instruction
      opcode = ident
      operands: List[Operand] = []

      while self.pos < self.length and self.code[self.pos] not in ("\n", ";"):
        op_leading = self._consume_whitespace_inline()

        if operands and self.pos < self.length and self.code[self.pos] == ",":
          op_leading += ","
          self._consume(1)
          op_leading += self._consume_whitespace_inline()

        if self.pos >= self.length or self.code[self.pos] in ("\n", ";"):
          # backtrack the whitespace so the comment or next node can have it as leading_trivia
          self.pos -= len(op_leading)
          break

        pre_pos = self.pos
        op_node = self._parse_operand()
        if self.pos == pre_pos:
          # Prevent infinite loops if operand parsing fails to advance
          self._consume(1)

        op_node.leading_trivia = op_leading
        operands.append(op_node)

      node = Instruction(opcode=opcode, operands=operands)
      node.leading_trivia = leading_trivia

      trailing = ""
      # peek ahead to see if there is a comment
      temp_pos = self.pos
      while temp_pos < self.length and self.code[temp_pos] in (" ", "\t", "\r"):
        temp_pos += 1

      if temp_pos < self.length and self.code[temp_pos] == ";":
        # leave whitespace for the comment
        pass
      else:
        trailing += self._consume_whitespace(newlines_only=True)
        if self.pos < self.length and self.code[self.pos] == "\n":
          trailing += "\n"
          self._consume(1)

      node.trailing_trivia = trailing
      nodes.append(node)

    return nodes

  def _parse_operand(self) -> Operand:
    """Parse an operand.

    Returns:
      Operand: The parsed operand.
    """
    ch = self.code[self.pos]

    if ch == "[":
      self._consume(1)
      self._consume_whitespace_inline()
      base_op = self._parse_operand()
      if not isinstance(base_op, (SGPR, VGPR)):
        raise ValueError("Invalid memory base")
      self._consume_whitespace_inline()

      offset = 0
      if self.pos < self.length and self.code[self.pos] in ("+", "-"):
        sign = -1 if self.code[self.pos] == "-" else 1
        self._consume(1)
        self._consume_whitespace_inline()
        off_op = self._parse_operand()
        if isinstance(off_op, Immediate):
          offset = sign * int(off_op.value)
        self._consume_whitespace_inline()

      if self.pos < self.length and self.code[self.pos] == "]":
        self._consume(1)

      return Memory(base=base_op, offset=offset)

    if ch.isdigit() or ch == "-":
      is_neg = False
      if ch == "-":
        is_neg = True
        self._consume(1)

      if self.pos < self.length and self.code[self.pos].isdigit():
        val_str = self._read_identifier(allow_hex=True, allow_dot=True)
        val_str = ("-" if is_neg else "") + val_str
        is_hex = "0x" in val_str.lower()
        val = float(val_str) if "." in val_str and not is_hex else int(val_str, 16 if is_hex else 10)
        return Immediate(value=val, is_hex=is_hex)
      else:
        return LabelRef(name="-")

    ident = self._read_identifier(allow_dot=True, allow_colon=True)

    if ident.startswith("s"):
      if self.pos < self.length and self.code[self.pos] == "[":
        self._consume(1)
        range_str = self._read_until("]")
        if self.pos < self.length:
          self._consume(1)  # ]
        parts = range_str.split(":")
        start_idx = int(parts[0])
        count = 1
        if len(parts) > 1:
          count = int(parts[1]) - start_idx + 1
        return SGPR(index=start_idx, count=count)
      elif ident[1:].isdigit():
        return SGPR(index=int(ident[1:]))

    if ident.startswith("v"):
      if self.pos < self.length and self.code[self.pos] == "[":
        self._consume(1)
        range_str = self._read_until("]")
        if self.pos < self.length:
          self._consume(1)  # ]
        parts = range_str.split(":")
        start_idx = int(parts[0])
        count = 1
        if len(parts) > 1:
          count = int(parts[1]) - start_idx + 1
        return VGPR(index=start_idx, count=count)
      elif ident[1:].isdigit():
        return VGPR(index=int(ident[1:]))

    if ident.lower() in ("glc", "slc", "off") or ident.lower().startswith("offset:"):
      return Modifier(name=ident)

    return LabelRef(name=ident)

  def _consume(self, count: int) -> None:
    """Consume count characters from the input stream.

    Args:
      count (int): Number of characters to consume.
    """
    self.pos += count

  def _consume_whitespace(self, newlines_only: bool = False) -> str:
    """Consume whitespace characters.

    Args:
      newlines_only (bool): If True, only consume newlines.

    Returns:
      str: The consumed whitespace.
    """
    start = self.pos
    while self.pos < self.length:
      ch = self.code[self.pos]
      if newlines_only:
        if ch not in (" ", "\t", "\r"):
          break
      else:
        if ch not in string.whitespace:
          break
      self.pos += 1
    return self.code[start : self.pos]

  def _consume_whitespace_inline(self) -> str:
    """Consume inline whitespace characters.

    Returns:
      str: The consumed inline whitespace.
    """
    start = self.pos
    while self.pos < self.length and self.code[self.pos] in (" ", "\t"):
      self.pos += 1
    return self.code[start : self.pos]

  def _read_identifier(self, allow_hex: bool = False, allow_dot: bool = False, allow_colon: bool = False) -> str:
    """Read an identifier from the input stream.

    Args:
      allow_hex (bool): If True, allow hex characters.
      allow_dot (bool): If True, allow dots.
      allow_colon (bool): If True, allow colons.

    Returns:
      str: The read identifier.
    """
    start = self.pos
    allowed = string.ascii_letters + string.digits + "_"
    if allow_hex:
      allowed += "xX"
    if allow_dot:
      allowed += "."
    if allow_colon:
      allowed += ":"

    while self.pos < self.length and self.code[self.pos] in allowed:
      self.pos += 1
    return self.code[start : self.pos]

  def _read_until(self, char: str) -> str:
    """Read characters until the specified character is encountered.

    Args:
      char (str): The character to stop reading at.

    Returns:
      str: The read characters.
    """
    start = self.pos
    while self.pos < self.length and self.code[self.pos] != char:
      self.pos += 1
    return self.code[start : self.pos]

  def _read_until_chars(self, chars: Tuple[str, ...]) -> str:
    """Read characters until one of the specified characters is encountered.

    Args:
      chars (Tuple[str, ...]): The characters to stop reading at.

    Returns:
      str: The read characters.
    """
    start = self.pos
    while self.pos < self.length and self.code[self.pos] not in chars:
      self.pos += 1
    return self.code[start : self.pos]
