"""MLIR Parser using Lark CST.

This module parses text-based MLIR code into the CST object model defined in `nodes.py`.
It utilizes Lark's Earley parser to handle explicit whitespace preservation without conflicts,
and transforms the resulting AST into strongly-typed CST nodes.
"""

import os
from typing import Any, List, Optional, Tuple, Union, cast

from lark import Lark, Token, Transformer, v_args

from ml_switcheroo.core.mlir.nodes import (
  ModuleNode,
  BlockNode,
  RegionNode,
  OperationNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  TriviaNode,
)


class MlirTransformer(Transformer[Token, Any]):
  """Transforms parsed MLIR AST into a CST with trivia."""

  @v_args(inline=True)
  def trivia(self, *tokens: Token) -> List[TriviaNode]:
    """Combine WS and COMMENT tokens into TriviaNodes.

    Args:
        tokens: The lexed tokens (whitespace, comments).

    Returns:
        List[TriviaNode]: A single TriviaNode containing the concatenated text.
    """
    content = "".join(str(t) for t in tokens)
    return [TriviaNode(content=content)]

  def module(self, children: List[Any]) -> ModuleNode:
    """Build a ModuleNode."""
    ops: List[OperationNode] = []
    leading: List[TriviaNode] = []
    trailing: List[TriviaNode] = []

    for c in children:
      if isinstance(c, list) and (len(c) == 0 or isinstance(c[0], TriviaNode)):
        if not ops:
          leading = c
        else:
          trailing = c
      elif isinstance(c, OperationNode):
        ops.append(c)

    # We create an implicit top-level block
    block = BlockNode(label="", operations=ops)
    return ModuleNode(body=block, leading_trivia=leading, trailing_trivia=trailing)

  def operation(self, children: List[Any]) -> OperationNode:
    """Build an OperationNode."""
    results_clause_dict: Optional[dict[str, Any]] = None
    name = ""
    name_trivia: List[TriviaNode] = []
    operands: List[ValueNode] = []
    attributes: List[AttributeNode] = []
    regions: List[RegionNode] = []
    result_types: List[TypeNode] = []
    trailing: List[TriviaNode] = []

    # Simple state machine to capture the tokens
    # Because of our grammar:
    # results_clause? (IDENTIFIER | STRING) trivia? operands_clause? attributes_clause? regions_clause? types_clause? trivia?
    found_name = False

    for c in children:
      if isinstance(c, dict) and "results" in c:
        results_clause_dict = c
      elif isinstance(c, str):
        name = c
        found_name = True
      elif isinstance(c, list):
        if not found_name:
          pass  # Should not happen based on grammar  # pragma: no cover
        elif not operands and not attributes and not regions and not result_types and not trailing:
          name_trivia = c
        else:
          trailing = c
      elif isinstance(c, tuple) and len(c) > 0 and c[0] == "operands":
        operands = c[1]
      elif isinstance(c, tuple) and len(c) > 0 and c[0] == "attributes":
        attributes = c[1]
      elif isinstance(c, tuple) and len(c) > 0 and c[0] == "regions":
        regions = c[1]
      elif isinstance(c, tuple) and len(c) > 0 and c[0] == "types":
        result_types = c[1]

    op = OperationNode(
      name=name,
      leading_trivia=[],
      name_trivia=name_trivia or [],
      trailing_trivia=trailing or [],
      results=results_clause_dict["results"] if results_clause_dict else [],
      operands=operands,
      attributes=attributes,
      regions=regions,
      result_types=result_types,
    )
    if results_clause_dict:
      op.leading_trivia = results_clause_dict["leading"]

    return op

  def results_clause(self, children: List[Any]) -> dict[str, Any]:
    """Build results clause."""
    results: List[ValueNode] = []
    trivia1: List[TriviaNode] = []
    trivia2: List[TriviaNode] = []
    for c in children:
      if isinstance(c, list) and isinstance(c[0], ValueNode):
        results = c
      elif isinstance(c, list) and isinstance(c[0], TriviaNode):
        if not trivia1:
          trivia1 = c
        else:
          trivia2 = c

    leading_trivia = []
    if results and results[0].leading_trivia:
      leading_trivia = results[0].leading_trivia  # pragma: no cover
      results[0].leading_trivia = []  # pragma: no cover

    if results and (trivia1 or trivia2):
      t = (trivia1 or []) + (trivia2 or [])
      results[-1].trailing_trivia.extend(t)

    return {"results": results, "leading": leading_trivia}

  def operands_clause(self, children: List[Any]) -> Tuple[str, List[ValueNode]]:
    """Build operands clause."""
    operands = []
    trivia = []
    for c in children:
      if isinstance(c, list) and len(c) > 0 and isinstance(c[0], ValueNode):
        operands = c
      elif isinstance(c, list) and len(c) > 0 and isinstance(c[0], TriviaNode):
        trivia = c
    if operands and trivia:
      operands[-1].trailing_trivia.extend(trivia)
    return ("operands", operands)

  def attributes_clause(self, children: List[Any]) -> Tuple[str, List[AttributeNode]]:
    """Build attributes clause."""
    attributes = []
    trivia = []
    for c in children:
      if isinstance(c, list) and len(c) > 0 and isinstance(c[0], AttributeNode):
        attributes = c
      elif isinstance(c, list) and len(c) > 0 and isinstance(c[0], TriviaNode):
        trivia = c
    if attributes and trivia:
      attributes[-1].trailing_trivia.extend(trivia)
    return ("attributes", attributes)

  def regions_clause(self, children: List[Any]) -> Tuple[str, List[RegionNode]]:
    """Build regions clause."""
    regions = []
    trivia = []
    for c in children:
      if isinstance(c, list) and len(c) > 0 and isinstance(c[0], RegionNode):
        regions = c
      elif isinstance(c, list) and len(c) > 0 and isinstance(c[0], TriviaNode):  # pragma: no cover
        trivia = c  # pragma: no cover
    if regions and trivia:
      regions[-1].trailing_trivia.extend(trivia)  # pragma: no cover
    return ("regions", regions)

  def types_clause(self, children: List[Any]) -> Tuple[str, List[TypeNode]]:
    """Build types clause."""
    types = []
    trivia = []
    for c in children:
      if isinstance(c, list) and len(c) > 0 and isinstance(c[0], TypeNode):
        types = c
      elif isinstance(c, list) and len(c) > 0 and isinstance(c[0], TriviaNode):
        trivia = c
    if types and trivia:
      types[0].leading_trivia = trivia + types[0].leading_trivia
    return ("types", types)

  def results(self, children: List[Any]) -> List[ValueNode]:
    """Build results list."""
    vals = []
    for c in children:
      if isinstance(c, ValueNode):
        vals.append(c)
      elif isinstance(c, list):  # trivia  # pragma: no cover
        if vals:  # pragma: no cover
          vals[-1].trailing_trivia.extend(c)  # pragma: no cover
    return vals

  def operands(self, children: List[Any]) -> List[ValueNode]:
    """Build operands list."""
    vals = []
    for c in children:
      if isinstance(c, ValueNode):
        vals.append(c)
      elif isinstance(c, list):  # trivia
        if vals:
          vals[-1].trailing_trivia.extend(c)
        else:
          # Leading trivia for the first operand
          pass  # Complex, handled by parent typically  # pragma: no cover
    return vals

  def attributes(self, children: List[Any]) -> List[AttributeNode]:
    """Build attributes list."""
    attrs = []
    for c in children:
      if isinstance(c, AttributeNode):
        attrs.append(c)
      elif isinstance(c, list):
        if attrs:
          attrs[-1].trailing_trivia.extend(c)
    return attrs

  def attribute(self, children: List[Any]) -> AttributeNode:
    """Build attribute node."""
    name = ""
    val: Union[str, List[str]] = ""
    trivia: List[TriviaNode] = []
    for c in children:
      if isinstance(c, str) and not name:
        name = c
      elif isinstance(c, list) and (len(c) == 0 or isinstance(c[0], TriviaNode)):
        trivia.extend(c)
      else:
        val = cast(Union[str, List[str]], c)
    attr = AttributeNode(name=name, value=val)
    attr.trailing_trivia = trivia
    return attr

  def array_attr(self, children: List[Any]) -> List[str]:
    """Build array attribute."""
    # Simplified string matching for now, preserving the CST values
    vals = []  # pragma: no cover
    for c in children:  # pragma: no cover
      if isinstance(c, str):  # pragma: no cover
        vals.append(c)  # pragma: no cover
    return vals  # pragma: no cover

  def regions(self, children: List[Any]) -> List[RegionNode]:
    """Build regions list."""
    regs = []
    for c in children:
      if isinstance(c, RegionNode):
        regs.append(c)
      elif isinstance(c, list):  # pragma: no cover
        if regs:  # pragma: no cover
          regs[-1].trailing_trivia.extend(c)  # pragma: no cover
    return regs

  def region(self, children: List[Any]) -> RegionNode:
    """Build region node."""
    blocks: List[BlockNode] = []
    leading: List[TriviaNode] = []
    trailing: List[TriviaNode] = []
    for c in children:
      if isinstance(c, BlockNode):
        blocks.append(c)
      elif isinstance(c, list):
        if not blocks:
          # It's leading trivia for the region content
          leading = c
        else:
          blocks[-1].trailing_trivia.extend(c)  # pragma: no cover
    # The last block's trailing trivia could also just be the region's trailing trivia
    # depending on spacing, but we handle it structurally for now.
    return RegionNode(blocks=blocks, leading_trivia=leading, trailing_trivia=trailing)

  def block(self, children: List[Any]) -> BlockNode:
    """Build block node."""
    label = ""
    args = []
    ops = []
    leading = []
    for c in children:
      if isinstance(c, dict) and "label" in c:
        label = c["label"]
        args = c["args"]
        leading = c["leading"]
      elif isinstance(c, OperationNode):
        ops.append(c)
    return BlockNode(label=label, arguments=args, operations=ops, leading_trivia=leading)

  def block_label_clause(self, children: List[Any]) -> dict[str, Any]:
    """Build block label clause."""
    name = ""
    args_clause: List[Tuple[ValueNode, TypeNode]] = []
    trivia1: List[TriviaNode] = []
    trivia2: List[TriviaNode] = []
    for c in children:
      if isinstance(c, str):
        name = c
      elif isinstance(c, list) and (len(c) == 0 or isinstance(c[0], TriviaNode)):
        if not args_clause:
          trivia1 = c
        else:
          trivia2 = c
      elif isinstance(c, list):
        # args clause is a list of tuples
        args_clause = c
    return {"label": name, "args": args_clause, "leading": (trivia1 or []) + (trivia2 or [])}

  def block_args_clause(self, children: List[Any]) -> List[Tuple[ValueNode, TypeNode]]:
    """Build block args clause."""
    for c in children:
      if isinstance(c, list) and len(c) > 0 and isinstance(c[0], tuple):
        return c
    return []  # pragma: no cover

  def block_arg_list(self, children: List[Any]) -> List[Tuple[ValueNode, TypeNode]]:
    """Build block arg list."""
    args = []
    for c in children:
      if isinstance(c, tuple):
        args.append(c)
    return args

  def block_arg(self, children: List[Any]) -> Tuple[ValueNode, TypeNode]:
    """Build block arg."""
    val = ""
    typ = ""
    trivia1 = []
    trivia2 = []
    for c in children:
      if isinstance(c, ValueNode):
        val = c.name
      elif isinstance(c, str):
        typ = c
      elif isinstance(c, list) and (len(c) == 0 or isinstance(c[0], TriviaNode)):
        if not typ:
          trivia1 = c
        else:
          trivia2 = c  # pragma: no cover
    v = ValueNode(name=str(val), trailing_trivia=trivia1 or [])
    t = TypeNode(body=str(typ), leading_trivia=trivia2 or [])
    return (v, t)

  def types(self, children: List[Any]) -> List[TypeNode]:
    """Build types list."""
    typs = []
    for c in children:
      if isinstance(c, str):
        typs.append(TypeNode(body=c))
      elif isinstance(c, list):
        if typs:
          typs[-1].trailing_trivia.extend(c)
    return typs

  # Terminal wrappers for simple types
  def IDENTIFIER(self, token: Token) -> str:
    """Wrap IDENTIFIER."""
    return str(token)

  def STRING(self, token: Token) -> str:
    """Wrap STRING."""
    return str(token)

  def NUMBER(self, token: Token) -> str:
    """Wrap NUMBER."""
    return str(token)

  def TYPE(self, token: Token) -> str:
    """Wrap TYPE."""
    return str(token)

  def VAL_ID(self, token: Token) -> ValueNode:
    """Wrap VAL_ID."""
    return ValueNode(name=str(token))

  def BLOCK_LABEL(self, token: Token) -> str:
    """Wrap BLOCK_LABEL."""
    return str(token)


class MlirParser:
  """Parses a stream of MLIR tokens into a Concrete Syntax Tree."""

  def __init__(self, text: str):
    """Initialize the parser.

    Args:
        text (str): The MLIR source code to parse.
    """
    self.text = text
    grammar_path = os.path.join(os.path.dirname(__file__), "grammar.lark")
    with open(grammar_path, "r", encoding="utf-8") as f:
      self.grammar = f.read()
    self.parser = Lark(self.grammar, parser="earley", maybe_placeholders=True)
    self.transformer = MlirTransformer()

  def parse(self) -> ModuleNode:
    """Top-level parsing entry point.

    Returns:
        ModuleNode: The root of the MLIR CST.
    """
    tree = self.parser.parse(self.text)
    from typing import cast

    return cast(ModuleNode, self.transformer.transform(tree))
