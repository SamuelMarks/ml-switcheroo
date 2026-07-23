"""SASS Parser Implementation.

This module provides the `SassParser`, a custom regex-free lexer/parser that converts
a stream of characters into a Structural AST defined in `nodes.py`.
"""

from typing import Tuple, Optional, List
import string

from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Memory,
  Operand,
  Predicate,
  Register,
  SassNode,
)

from dataclasses import dataclass


@dataclass
class LabelRef(Operand):
  """Execute implementation detail."""

  name: str

  def __str__(self) -> str:
    """Execute implementation detail."""
    return self.name


class SassParser:
  """Custom LLVM-MC style parser for NVIDIA SASS."""

  def __init__(self, code: str) -> None:
    """Initialize the parser.

    Args:
        code (str): The raw SASS source string.
    """
    self.code = code
    self.pos = 0
    self.length = len(code)

  def parse(self) -> List[SassNode]:
    """Parses the entire code block.

    Returns:
        List[SassNode]: A list of AST nodes.
    """
    nodes: List[SassNode] = []

    if not self.code.strip() or self.code == ";":
      return []

    while self.pos < self.length:
      leading_trivia = self._consume_whitespace()

      if self.pos >= self.length:
        if leading_trivia and nodes:
          nodes[-1].trailing_trivia += leading_trivia
        break

      ch = self.code[self.pos]

      if ch == "/" and self._peek() == "/":
        # Comment
        self._consume(2)
        text = self._read_until("\n").strip()
        comment_node = Comment(text=text)
        comment_node.leading_trivia = leading_trivia
        comment_node.trailing_trivia = self._consume_whitespace(newlines_only=True)
        nodes.append(comment_node)
        continue

      if ch == ";":
        # Empty statement / trailing semicolon from previous node without newline
        self._consume(1)
        if nodes:
          nodes[-1].trailing_trivia += leading_trivia + ";"
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
        if self.pos < self.length and self.code[self.pos] == ";":
          self._consume(1)
        dir_node.trailing_trivia = self._consume_whitespace(newlines_only=True)
        if self.pos < self.length and self.code[self.pos] == "\n":
          dir_node.trailing_trivia += "\n"
          self._consume(1)
        nodes.append(dir_node)
        continue

      # Instructions or Labels
      predicate = None
      if ch == "@":
        self._consume(1)
        pred_name = self._read_identifier(allow_bang=True)
        negated = pred_name.startswith("!")
        if negated:
          pred_name = pred_name[1:]
        predicate = Predicate(name=pred_name, negated=negated)
        predicate.leading_trivia = leading_trivia
        leading_trivia = self._consume_whitespace()

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
      pre_instr_pos = self.pos
      opcode = ident
      operands: List[Operand] = []

      while self.pos < self.length and self.code[self.pos] not in ("\n", ";", "/"):
        op_leading = self._consume_whitespace_inline()

        if operands and self.pos < self.length and self.code[self.pos] == ",":
          op_leading += ","
          self._consume(1)
          op_leading += self._consume_whitespace_inline()

        if self.pos >= self.length or self.code[self.pos] in ("\n", ";", "/"):
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

      node = Instruction(opcode=opcode, operands=operands, predicate=predicate)
      node.leading_trivia = leading_trivia

      trailing = ""
      if self.pos < self.length and self.code[self.pos] == ";":
        trailing += ";"
        self._consume(1)

      # peek ahead to see if there is a comment
      temp_pos = self.pos
      while temp_pos < self.length and self.code[temp_pos] in (" ", "\t", "\r"):
        temp_pos += 1

      if (
        temp_pos < self.length
        and self.code[temp_pos] == "/"
        and temp_pos + 1 < self.length
        and self.code[temp_pos + 1] == "/"
      ):
        # leave whitespace for the comment
        pass
      else:
        trailing += self._consume_whitespace(newlines_only=True)
        if self.pos < self.length and self.code[self.pos] == "\n":
          trailing += "\n"
          self._consume(1)

      if self.pos == pre_instr_pos and opcode == "":
        # Prevent infinite loop if entire loop consumed nothing
        trailing += self.code[self.pos]
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
      # Memory block: [R1 + 0x4]
      self._consume(1)
      self._consume_whitespace_inline()
      base_str = self._read_identifier()
      self._consume_whitespace_inline()
      offset = None
      if self.pos < self.length and self.code[self.pos] == "+":
        self._consume(1)
        self._consume_whitespace_inline()
        off_str = self._read_identifier()
        offset = int(off_str, 16) if off_str.lower().startswith("0x") else int(off_str)

      while self.pos < self.length and self.code[self.pos] != "]":
        self._consume(1)
      if self.pos < self.length:
        self._consume(1)  # consume ]
      return Memory(base=Register(name=base_str), offset=offset)

    if ch == "c" and self._peek() == "[":
      # Constant memory: c[0x0][0x4]
      self._consume(2)
      bank_str = self._read_identifier()
      while self.pos < self.length and self.code[self.pos] != "]":
        self._consume(1)
      if self.pos < self.length:
        self._consume(1)  # ]

      offset = None
      if self.pos < self.length and self.code[self.pos] == "[":
        self._consume(1)
        off_str = self._read_identifier()
        offset = int(off_str, 16) if off_str.lower().startswith("0x") else int(off_str)
        while self.pos < self.length and self.code[self.pos] != "]":
          self._consume(1)
        if self.pos < self.length:
          self._consume(1)  # ]
      else:
        offset = None  # Default offset for c[X] is None
      return Memory(base=f"c[{bank_str}]", offset=offset)

    if ch.isdigit() or ch == "-":
      # Immediate or Negated Register/Absolute
      is_neg = False
      if ch == "-":
        is_neg = True
        self._consume(1)

      if self.pos < self.length and self.code[self.pos] == "|":
        self._consume(1)
        reg_name = self._read_identifier()
        if self.pos < self.length and self.code[self.pos] == "|":
          self._consume(1)
        return Register(name=reg_name, negated=is_neg, absolute=True)

      if self.pos < self.length and self.code[self.pos].isdigit():
        val_str = self._read_identifier(allow_hex=True, allow_dot=True)
        val_str = ("-" if is_neg else "") + val_str
        is_hex = "0x" in val_str.lower()
        val = float(val_str) if "." in val_str and not is_hex else int(val_str, 16 if is_hex else 10)
        return Immediate(value=val, is_hex=is_hex)
      else:
        # Negated register
        reg_name = self._read_identifier()
        return Register(name=reg_name, negated=True)

    if ch == "|":
      # Absolute register
      self._consume(1)
      is_neg = False
      if self.pos < self.length and self.code[self.pos] == "-":
        is_neg = True
        self._consume(1)
      reg_name = self._read_identifier()
      if self.pos < self.length and self.code[self.pos] == "|":
        self._consume(1)
      return Register(name=reg_name, negated=is_neg, absolute=True)

    if ch == "!":
      self._consume(1)
      return Predicate(name=self._read_identifier(), negated=True)

    if ch == "@":
      self._consume(1)
      return Predicate(name=self._read_identifier(), negated=False)

    ident = self._read_identifier()
    if self.pos < self.length and self.code[self.pos] == ":":
      self._consume(1)  # Consume trailing colon for label refs

    if ident.startswith("P") or ident == "PT":
      return Predicate(name=ident)

    if ident.startswith("L_"):
      return LabelRef(name=ident)

    return Register(name=ident)

  def _peek(self) -> Optional[str]:
    """Peek at the next character.

    Returns:
      Optional[str]: The next character if available.
    """
    if self.pos + 1 < self.length:
      return self.code[self.pos + 1]
    return None

  def _consume(self, count: int) -> None:
    """Consume count characters.

    Args:
      count (int): Number of characters.
    """
    self.pos += count

  def _consume_whitespace(self, newlines_only: bool = False) -> str:
    """Consume whitespace.

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
    """Consume inline whitespace.

    Returns:
      str: The consumed inline whitespace.
    """
    start = self.pos
    while self.pos < self.length and self.code[self.pos] in (" ", "\t"):
      self.pos += 1
    return self.code[start : self.pos]

  def _read_identifier(self, allow_bang: bool = False, allow_hex: bool = False, allow_dot: bool = False) -> str:
    """Read an identifier.

    Args:
      allow_bang (bool): Allow exclamation mark.
      allow_hex (bool): Allow hex chars.
      allow_dot (bool): Allow dots.

    Returns:
      str: The read identifier.
    """
    start = self.pos
    allowed = string.ascii_letters + string.digits + "_"
    if allow_bang:
      allowed += "!"
    if allow_hex:
      allowed += "xX"
    if allow_dot:
      allowed += "."

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
