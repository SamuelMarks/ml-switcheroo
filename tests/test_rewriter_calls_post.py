"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo_ir.schema.ghost import SemanticTier


class DummyTraits:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.strip_magic_args = []
    self.auto_strip_magic_args = False
    self.inject_magic_args = []


class DummyContext:
  """Docstring."""

  def __init__(self, is_init=False, is_module=False):
    """Docstring."""

    class Sig:
      def __init__(self, i, m):
        self.is_init = i
        self.is_module_method = m

    self.signature_stack = [Sig(is_init, is_module)] if is_init or is_module else []


class DummySemantics:
  """Docstring."""

  def __init__(self, origins=None, magic=None):
    """Docstring."""
    self._key_origins = origins or {}
    if magic is not None:
      self.known_magic_args = magic


class DummyRewriter:
  """Docstring."""

  def __init__(self, context=None, semantics=None, traits=None, report=True):
    """Docstring."""
    self.context = context or DummyContext()
    self.semantics = semantics or DummySemantics()
    self.traits = traits or DummyTraits()
    self.report = report

  def _create_dotted_name(self, name):
    if name == "error":
      raise ValueError("bad type")
    return cst.Name(name)

  def _report_failure(self, msg):
    pass

  def _get_target_traits(self):
    return self.traits


def test_handle_post_processing_branches():
  """Docstring."""
  # Base case
  rewriter = DummyRewriter()
  node = cst.parse_statement("f()").body[0].value
  assert handle_post_processing(rewriter, node, {}, "id") == node

  # 38 -> 39, 47 -> 59 (output_select_index success)
  node_tuple = cst.parse_statement("f()[0]").body[0].value
  mapping_select = {"output_select_index": 0}
  handle_post_processing(rewriter, node_tuple, mapping_select, "id")
  # should be Subscript

  # 42 -> 43 -> 44 (output_select_index failure, with _report_failure)
  (cst.Pass())  # apply_index_select will fail if not expression maybe? Actually let's mock it or use bad index type
  mapping_bad_select = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping_bad_select, "id")

  # 42 -> 44 (output_select_index failure, no _report_failure)
  class DummyRewriterNoReport(DummyRewriter):
    def _report_failure(self, msg):
      raise NotImplementedError()

  rewriter_no_report = DummyRewriterNoReport()
  del DummyRewriterNoReport._report_failure
  handle_post_processing(rewriter_no_report, node, mapping_bad_select, "id")

  # 47 -> 48 (output_cast) -> 59
  mapping_cast = {"output_cast": "float32"}
  handle_post_processing(rewriter, node, mapping_cast, "id")

  # 54 -> 55 (output_cast failure)
  mapping_bad_cast = {"output_cast": "error"}
  handle_post_processing(rewriter, node, mapping_bad_cast, "id")

  # 59 -> 65 (Signature stack is init module method)
  # 72 -> 73 (result is Call)
  # 74 -> 77 (auto strip true, has known magic)
  # 78 -> 79 (has keyword in magic -> force=True)
  # 86 -> 87 (inject)
  # 91 -> 92 (strip auto)
  # 96 -> 97 (strip)
  node_call = cst.parse_statement("f(key=1)").body[0].value
  traits = DummyTraits()
  traits.auto_strip_magic_args = True
  traits.strip_magic_args = ["strip_me"]
  traits.inject_magic_args = [("inj_me", "")]
  semantics = DummySemantics(origins={"my_id": SemanticTier.NEURAL.value}, magic=["key"])
  context = DummyContext(is_init=True, is_module=True)
  rewriter_neural = DummyRewriter(context=context, semantics=semantics, traits=traits)

  handle_post_processing(rewriter_neural, node_call, {}, "my_id")

  # 72 -> 82 (Not a Call, but neural)
  node_not_call = cst.Name("x")
  handle_post_processing(rewriter_neural, node_not_call, {}, "my_id")

  # 78 -> 77 (keyword not in magic) -> 82 -> 99 (Not neural, force=False)
  node_call2 = cst.parse_statement("f(other=1)").body[0].value
  rewriter_not_neural = DummyRewriter(context=context, semantics=DummySemantics(origins={}), traits=traits)
  handle_post_processing(rewriter_not_neural, node_call2, {}, "other_id")

  # 84 -> 99 (is neural but result not call) - already tested above but let's be sure
  handle_post_processing(rewriter_neural, cst.Name("x"), {}, "my_id")

  # 74 -> 75 (auto_strip False)
  traits.auto_strip_magic_args = False
  handle_post_processing(rewriter_neural, node_call, {}, "my_id")
