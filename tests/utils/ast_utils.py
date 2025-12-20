"""
ast_utils, a bunch of helpers for converting input into ast.* input_str
"""

from ast import AST


class Undefined:
  """Null class"""


def cmp_ast(node0, node1):
  """
  Compare if two nodes are equal. Verbatim stolen from `meta.asttools`.

  :param node0: First node
  :type node0: ```Union[AST, List[AST], Tuple[AST]]```

  :param node1: Second node
  :type node1: ```Union[AST, List[AST], Tuple[AST]]```

  :return: Whether they are equal (recursive)
  :rtype: ```bool```
  """

  if type(node0) is not type(node1):
    return False

  if isinstance(node0, (list, tuple)):
    if len(node0) != len(node1):
      return False

    for left, right in zip(node0, node1):
      if not cmp_ast(left, right):
        return False

  elif isinstance(node0, AST):
    for field in node0._fields:
      left = getattr(node0, field, Undefined)
      right = getattr(node1, field, Undefined)

      if not cmp_ast(left, right):
        return False
  else:
    return node0 == node1

  return True
