"""Test suite for the Visualizer Missing module."""


def test_visualizer_exceptions():
  """Verifies the behavior of visualizer exceptions."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()

  class BadNode(cst.CSTNode):
    """Test suite for the Bad Node component."""

    def _codegen_impl(self, state):
      """Helper to  codegen impl."""
      raise Exception("fail")

    def _visit_and_replace_children(self, v):
      """Helper to  visit and replace children."""
      return self

  assert "<BadNode>" in gen._node_to_str(BadNode())
  with __import__("unittest.mock").mock.patch.object(gen, "_node_to_str", side_effect=Exception("fail")):
    call_node = cst.Call(func=cst.Name("foo"))
    gen.visit_Call(call_node)
  with __import__("unittest.mock").mock.patch.object(gen, "_node_to_str", side_effect=Exception("fail")):
    arg_node = cst.Arg(value=cst.Name("foo"))
    gen.visit_Arg(arg_node)
  gen.stack.clear()
  gen.leave_Assign(cst.Assign(targets=[cst.AssignTarget(cst.Name("a"))], value=cst.Pass()))
  gen.stack.clear()
  gen.visit_SimpleString(cst.SimpleString('""'))


def test_visualizer_more_nodes():
  """Verifies the behavior of visualizer more nodes."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()
  cls_node = cst.ClassDef(name=cst.Name("Foo"), body=cst.IndentedBlock([]))
  gen.visit_ClassDef(cls_node)
  gen.leave_ClassDef(cls_node)
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("foo")), cst.ImportAlias(name=cst.Name("bar"))])
  gen.visit_Import(imp_node)
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
  imp_from_star = cst.ImportFrom(module=cst.Name("foo"), names=cst.ImportStar())
  gen.visit_ImportFrom(imp_from_star)


def test_visualizer_more_fallbacks():
  """Verifies the behavior of visualizer more fallbacks."""
  import libcst as cst
  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen = MermaidGenerator()
  call = cst.Call(func=cst.Call(func=cst.Name("a")))
  gen.visit_Call(call)

  class DummyImportAlias(cst.ImportAlias):
    """Dummy Import Alias class for testing purposes."""

    pass

  imp = cst.Import(names=[DummyImportAlias(name=cst.Attribute(cst.Name("a"), cst.Name("b")))])
  gen.visit_Import(imp)
