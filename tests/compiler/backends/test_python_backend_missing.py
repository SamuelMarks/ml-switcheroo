"""Tests."""

import libcst as cst
import typing
from ml_switcheroo.core.compiler.backends.python import PythonBackend, ClassBodyReplacer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.ir import LogicalEdge


def test_class_body_replacer_else_branch() -> None:
  """Test function."""
  init_stmt = typing.cast(cst.FunctionDef, cst.parse_statement("def __init__(self): pass"))
  forward_stmt = typing.cast(cst.FunctionDef, cst.parse_statement("def forward(self): pass"))
  replacer = ClassBodyReplacer("X", init_stmt, forward_stmt)
  mod: cst.Module = cst.parse_module(
    "class X:\n    def __init__(self):\n        pass\n    def other(self):\n        print(1)\n        pass\n"
  )
  res: cst.Module = mod.visit(replacer)
  assert "print(1)" in res.code

  class DummyNode(cst.ClassDef):
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      super().__init__(name=cst.Name("Dummy"), body=cst.IndentedBlock(body=[]))

  d = DummyNode()
  assert replacer.leave_ClassDef(d, d) is d


def test_python_backend_imports() -> None:
  """Test function."""
  pass


def test_python_backend_compile_forward_pass_no_stmts() -> None:
  """Test function."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph("Test")
  graph.nodes.append(LogicalNode("input_0", "Input"))
  res: str = backend.compile(graph)
  assert "def forward" in res


def test_python_backend_forward_pass_abstract_resolution() -> None:
  """Test function."""
  semantics = SemanticsManager()

  original_resolve = semantics.resolve_variant
  original_get = semantics.get_definition

  def mock_resolve(api: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Test function."""
    if api == "my_func":
      return {"api": "resolved.my_func"}
    if api == "my_abstract":
      return {"api": "resolved.my_abstract"}
    return original_resolve(api, fw)

  def mock_get(api: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Test function."""
    if api == "func_concrete_func":
      return ("my_abstract", {})
    return original_get(api)

  semantics.resolve_variant = mock_resolve
  semantics.get_definition = mock_get

  backend = PythonBackend(framework="torch", semantics=semantics)

  # We must patch backend._is_stateful_layer because it decides functional vs object state
  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Test function."""
    return False

  backend._is_stateful_layer = mock_is_stateful_layer

  graph = LogicalGraph("Test")
  n0 = LogicalNode("n0", "Input")

  n1 = LogicalNode("n1", "my_func")
  n2 = LogicalNode("n2", "func_concrete_func")

  graph.nodes.extend([n0, n1, n2])
  graph.edges.extend([LogicalEdge("n0", "n1"), LogicalEdge("n1", "n2")])
  code: str = backend.compile(graph)
  assert "resolved.my_func" in code
  assert "resolved.my_abstract" in code
