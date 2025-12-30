"""
MLIR Recursive Descent Parser.

This module parses text-based MLIR code into the CST object model defined in `nodes.py`.
It is designed to preserve trivia (comments/whitespace) to support high-fidelity
round-trip transformations.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator

from ml_switcheroo.core.mlir.nodes import (
  ModuleNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  TriviaNode,
)
from ml_switcheroo.core.mlir.tokens import TokenKind, Symbol


@dataclass
class Token:
  kind: str
  text: str
  line: int
  col: int


class Tokenizer:
  PATTERN_DEFS = [
    (TokenKind.COMMENT, r"//[^\n]*"),
    (TokenKind.STRING, r'"(?:[^"\\]|\\.)*"'),
    (TokenKind.REGION_TYPE, r"!sw\.type<[^>]+>"),
    (TokenKind.TYPE, r"![a-zA-Z_0-9\.<>]+|tensor<[^>]+>|[iuf]\d+|index|none"),
    (TokenKind.VAL_ID, r"%[a-zA-Z_0-9]+|%\d+"),
    (TokenKind.SYM_ID, r"@[a-zA-Z_0-9]+"),
    (TokenKind.BLOCK_LABEL, r"\^[a-zA-Z_0-9]+"),
    (TokenKind.ARROW, r"->"),
    (TokenKind.SYMBOL, r"[(){}\[\],:=]"),
    (TokenKind.IDENTIFIER, r"[a-zA-Z_][a-zA-Z0-9_$.]*"),
    (TokenKind.NUMBER, r"-?\d+(?:\.\d+)?"),
    (TokenKind.NEWLINE, r"\n"),
    (TokenKind.WHITESPACE, r"[ \t]+"),
    (TokenKind.MISMATCH, r"."),
  ]

  _REGEX = re.compile("|".join(f"(?P<{kind.value}>{pattern})" for kind, pattern in PATTERN_DEFS))

  def __init__(self, text: str):
    self.text = text

  def tokenize(self) -> Generator[Token, None, None]:
    line_num = 1
    line_start = 0
    for mo in self._REGEX.finditer(self.text):
      kind_str = mo.lastgroup
      value = mo.group()
      col = mo.start() - line_start

      try:
        kind = TokenKind(kind_str)
      except ValueError:
        kind = kind_str

      if kind == TokenKind.NEWLINE:
        yield Token(kind, value, line_num, col)
        line_num += 1
        line_start = mo.end()
      elif kind == TokenKind.MISMATCH:
        raise ValueError(f"Unexpected character {value!r} on line {line_num}:{col}")
      else:
        yield Token(kind, value, line_num, col)
    yield Token(TokenKind.EOF, "", line_num, 0)


class MlirParser:
  def __init__(self, text: str):
    self.tokenizer = Tokenizer(text)
    self.tokens = list(self.tokenizer.tokenize())
    self.pos = 0
    self.trivia_buffer: List[TriviaNode] = []

  def peek(self, offset: int = 0) -> Token:
    idx = self.pos + offset
    if idx >= len(self.tokens):
      return self.tokens[-1]
    return self.tokens[idx]

  def consume(self) -> Token:
    token = self.peek()
    self.pos += 1
    return token

  def match(self, kind: str) -> bool:
    tk = self.peek()
    if tk.kind == kind:
      return True
    if tk.kind == TokenKind.SYMBOL:
      if tk.text == kind:
        return True
    return False

  def expect(self, kind: str) -> Token:
    if not self.match(kind):
      cur = self.peek()
      raise SyntaxError(f"Expected {kind}, got {cur.kind} ('{cur.text}')")
    return self.consume()

  def _flush_trivia(self) -> List[TriviaNode]:
    t = self.trivia_buffer
    self.trivia_buffer = []
    return t

  def _absorb_trivia(self):
    while True:
      tk = self.peek()
      if tk.kind == TokenKind.EOF:
        break
      if tk.kind in (TokenKind.WHITESPACE, TokenKind.COMMENT, TokenKind.NEWLINE):
        self.consume()
        kmap = {"WHITESPACE": "whitespace", "NEWLINE": "newline", "COMMENT": "comment"}
        k_str = tk.kind.value if hasattr(tk.kind, "value") else str(tk.kind)
        self.trivia_buffer.append(TriviaNode(tk.text, kind=kmap.get(k_str, "whitespace")))
      else:
        break

  def parse(self) -> ModuleNode:
    return ModuleNode(body=self.parse_block(is_top_level=True))

  def parse_block(self, is_top_level: bool = False) -> BlockNode:
    label = ""
    arguments = []
    self._absorb_trivia()
    leading = self._flush_trivia()

    if not is_top_level and self.match(TokenKind.BLOCK_LABEL):
      label = self.consume().text
      self._absorb_trivia()
      if self.match(Symbol.LPAREN):
        self.consume()
        while not self.match(Symbol.RPAREN):
          self._absorb_trivia()
          self._flush_trivia()  # Fix: Discard whitespace trivia inside arg list to prevent leaking

          if self.match(TokenKind.VAL_ID):
            vn = self.consume().text
            self._absorb_trivia()
            self._flush_trivia()  # Fix

            if self.match(Symbol.COLON):
              self.consume()
              self._absorb_trivia()
              if self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):
                arguments.append((ValueNode(vn), TypeNode(self.consume().text)))

          self._absorb_trivia()
          self._flush_trivia()  # Fix

          if self.match(Symbol.COMMA):
            self.consume()
          else:
            break
        self.expect(Symbol.RPAREN)
      self._absorb_trivia()
      if self.match(Symbol.COLON):
        self.consume()

    operations = []
    while True:
      self._absorb_trivia()
      tk = self.peek()
      if tk.kind == TokenKind.EOF:
        break
      if tk.text == Symbol.RBRACE and not is_top_level:
        break
      if tk.kind == TokenKind.BLOCK_LABEL:
        break

      op_trivia = self._flush_trivia()

      op = self.parse_operation()
      if op:
        op.leading_trivia = op_trivia + op.leading_trivia
        operations.append(op)
      else:
        if tk.kind == TokenKind.EOF or tk.text == Symbol.RBRACE or tk.kind == TokenKind.BLOCK_LABEL:
          break
        raise SyntaxError(f"Unexpected token {tk.kind} ('{tk.text}') where Op expected")

    return BlockNode(label=label, arguments=arguments, operations=operations, leading_trivia=leading)

  def _is_region_start(self) -> bool:
    offset = 1
    while True:
      t = self.peek(offset)
      if t.kind in (TokenKind.WHITESPACE, TokenKind.NEWLINE, TokenKind.COMMENT):
        offset += 1
        continue
      break

    if t.kind == TokenKind.BLOCK_LABEL:
      return True
    if t.kind == TokenKind.VAL_ID:
      return True
    if t.text == Symbol.RBRACE:
      return True

    if t.kind in (TokenKind.STRING, TokenKind.IDENTIFIER):
      scan_off = offset + 1
      while True:
        t2 = self.peek(scan_off)
        if t2.kind in (TokenKind.WHITESPACE, TokenKind.NEWLINE, TokenKind.COMMENT):
          scan_off += 1
          continue
        break

      if t2.text == Symbol.EQUAL:
        return False

      return True

    return False

  def parse_operation(self) -> Optional[OperationNode]:
    results = []
    lh = 0
    eq_found = False
    while True:
      tk = self.peek(lh)
      if tk.kind in (TokenKind.EOF, TokenKind.NEWLINE, TokenKind.BLOCK_LABEL) or tk.text in [
        Symbol.LBRACE,
        Symbol.RBRACE,
      ]:
        break
      if tk.text == Symbol.EQUAL:
        eq_found = True
        break
      if lh > 20:
        break
      lh += 1

    if eq_found:
      while self.peek().text != Symbol.EQUAL:
        start_pos = self.pos
        if self.match(TokenKind.VAL_ID):
          results.append(ValueNode(self.consume().text))
        elif self.match(Symbol.COMMA):
          self.consume()

        self._absorb_trivia()
        if self.peek().text == Symbol.EQUAL:
          break
        if self.pos == start_pos:
          raise SyntaxError(f"Stuck parsing results at {self.peek().text}")

      self.consume()

      self._absorb_trivia()
      self._flush_trivia()

    op_name = ""
    if self.match(TokenKind.STRING) or self.match(TokenKind.IDENTIFIER):
      op_name = self.consume().text
      while self.peek().text == ".":
        self.consume()
        if self.match(TokenKind.IDENTIFIER):
          op_name += "." + self.consume().text
    else:
      return None

    self._absorb_trivia()
    name_trivia = self._flush_trivia()

    operands = []
    if self.peek().text == Symbol.LPAREN:
      self.consume()
      while not self.match(Symbol.RPAREN):
        self._absorb_trivia()
        if self.match(TokenKind.VAL_ID) or self.match(TokenKind.SYM_ID):
          operands.append(ValueNode(self.consume().text))
        elif self.match(Symbol.COMMA):
          self.consume()
        else:
          break
      self.expect(Symbol.RPAREN)

    self._absorb_trivia()
    self._flush_trivia()

    attributes = []
    if self.match(Symbol.LBRACE):
      if not self._is_region_start():
        self.consume()
        while not self.match(Symbol.RBRACE):
          self._absorb_trivia()
          if self.peek().kind == TokenKind.EOF:
            break
          if self.match(Symbol.RBRACE):
            break

          if self.match(TokenKind.IDENTIFIER) or self.match(TokenKind.STRING):
            key = self.consume().text
            self._absorb_trivia()
            if self.match(Symbol.EQUAL):
              self.consume()
              self._absorb_trivia()
              val_s = []
              while True:
                txt = self.peek().text
                if txt in [Symbol.COMMA, Symbol.RBRACE, Symbol.COLON]:
                  break
                if self.peek().kind == TokenKind.EOF:
                  break

                tk = self.consume()
                val_s.append(tk.text)
                if self.peek().kind == TokenKind.WHITESPACE:
                  self.consume()

              val_str = "".join(val_s).strip()

              tp = None
              if self.match(Symbol.COLON):
                self.consume()
                self._absorb_trivia()
                if self.match(TokenKind.TYPE):
                  tp = self.consume().text
              attributes.append(AttributeNode(key, val_str, tp))
          self._absorb_trivia()
          if self.match(Symbol.COMMA):
            self.consume()
        self.expect(Symbol.RBRACE)

    self._absorb_trivia()
    self._flush_trivia()

    regions = []
    if self.peek().text == Symbol.LBRACE:
      if self._is_region_start():
        regions.append(self.parse_region())

    self._absorb_trivia()
    res_types = []
    if self.match(Symbol.COLON):
      self._flush_trivia()
      self.consume()
      self._absorb_trivia()
      if self.match(Symbol.LPAREN):
        self.consume()
        while not self.match(Symbol.RPAREN):
          self._absorb_trivia()
          if self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):
            self._flush_trivia()
            res_types.append(TypeNode(self.consume().text))
          if self.match(Symbol.COMMA):
            self.consume()
        self.consume()
      elif self.match(TokenKind.TYPE) or self.match(TokenKind.REGION_TYPE):
        self._flush_trivia()
        res_types.append(TypeNode(self.consume().text))

    if self.match(TokenKind.ARROW):
      self.consume()
      self._absorb_trivia()
      pass

    self._absorb_trivia()
    trailing = self._flush_trivia()

    return OperationNode(
      op_name, results, operands, attributes, regions, res_types, name_trivia=name_trivia, trailing_trivia=trailing
    )

  def parse_region(self) -> RegionNode:
    self.expect(Symbol.LBRACE)
    blocks = []
    while True:
      self._absorb_trivia()
      if self.peek().kind == TokenKind.EOF:
        break
      if self.match(Symbol.RBRACE):
        break

      blk = self.parse_block(is_top_level=False)
      blocks.append(blk)

      if not blk.operations and not blk.label:
        if self.match(Symbol.RBRACE):
          break
        self.consume()

      if self.match(Symbol.RBRACE):
        break

    if self.match(Symbol.RBRACE):
      self.consume()
    return RegionNode(blocks=blocks)
