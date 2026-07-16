"""Auto-generated doc."""


def test_visualizer_exceptions():
  """Auto-generated doc."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()

  # 137-138
  class BadNode(cst.CSTNode):
    """Auto-generated doc."""

    def _codegen_impl(self, state):
      """Auto-generated doc."""
      raise Exception("fail")

    def _visit_and_replace_children(self, v):
      """Auto-generated doc."""
      return self

  assert "<BadNode>" in gen._node_to_str(BadNode())

  # 184-185: exception inside visit_Call
  # We can just make the _node_to_str fail
  with __import__("unittest.mock").mock.patch.object(gen, "_node_to_str", side_effect=Exception("fail")):
    call_node = cst.Call(func=cst.Name("foo"))
    gen.visit_Call(call_node)

  # 217-218: exception inside visit_Arg trying to format value
  with __import__("unittest.mock").mock.patch.object(gen, "_node_to_str", side_effect=Exception("fail")):
    arg_node = cst.Arg(value=cst.Name("foo"))
    gen.visit_Arg(arg_node)

  # 247: leave_Assign popping from empty stack
  gen.stack.clear()
  gen.leave_Assign(cst.Assign(targets=[cst.AssignTarget(cst.Name("a"))], value=cst.Pass()))

  # 255: visit_SimpleString pushing to empty stack (ignoring edge case)
  gen.stack.clear()
  gen.visit_SimpleString(cst.SimpleString('""'))


def test_visualizer_more_nodes():
  """Auto-generated doc."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()

  # ClassDef
  cls_node = cst.ClassDef(name=cst.Name("Foo"), body=cst.IndentedBlock([]))
  gen.visit_ClassDef(cls_node)
  gen.leave_ClassDef(cls_node)

  # Import
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("foo")), cst.ImportAlias(name=cst.Name("bar"))])
  gen.visit_Import(imp_node)

  # ImportFrom
  imp_from_node = cst.ImportFrom(
    module=cst.Name("foo"),
    names=[
      cst.ImportAlias(name=cst.Name("a")),
      cst.ImportAlias(name=cst.Name("b")),
      cst.ImportAlias(name=cst.Name("c")),
      cst.ImportAlias(name=cst.Name("d")),
    ],
  )
  gen.visit_ImportFrom(imp_from_node)

  # ImportFrom Star
  imp_from_star = cst.ImportFrom(module=cst.Name("foo"), names=cst.ImportStar())
  gen.visit_ImportFrom(imp_from_star)


def test_visualizer_more_fallbacks():
  """Auto-generated doc."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()

  # 180: call with fallback name inside try block
  call = cst.Call(func=cst.Call(func=cst.Name("a")))
  gen.visit_Call(call)

  # 238: Import names empty
  class DummyImportAlias(cst.ImportAlias):
    """Auto-generated doc."""

    pass

  imp = cst.Import(names=[DummyImportAlias(name=cst.Attribute(cst.Name("a"), cst.Name("b")))])
  gen.visit_Import(imp)
