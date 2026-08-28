"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo_ir.schema.ghost import SemanticTier
from typing import List, Dict, Any, Tuple, Optional


class DummyTraits:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.strip_magic_args: List[str] = []
    self.auto_strip_magic_args: bool = False
    self.inject_magic_args: List[Tuple[str, str]] = []


class DummyContext:
  """Docstring."""

  def __init__(self, is_init: bool = False, is_module: bool = False) -> None:
    """Docstring."""

    class Sig:
      """Docstring."""

      def __init__(self, i: bool, m: bool) -> None:
        """Docstring."""
        self.is_init = i
        self.is_module_method = m

    self.signature_stack: List[Sig] = [Sig(is_init, is_module)] if is_init or is_module else []


class DummySemantics:
  """Docstring."""

  def __init__(self, origins: Optional[Dict[str, str]] = None, magic: Optional[List[str]] = None) -> None:
    """Docstring."""
    self._key_origins: Dict[str, str] = origins or {}
    if magic is not None:
      self.known_magic_args: List[str] = magic


class DummyRewriter:
  """Docstring."""

  def __init__(
    self,
    context: Optional[DummyContext] = None,
    semantics: Optional[DummySemantics] = None,
    traits: Optional[DummyTraits] = None,
    report: bool = True,
  ) -> None:
    """Docstring."""
    self.context: DummyContext = context or DummyContext()
    self.semantics: DummySemantics = semantics or DummySemantics()
    self.traits: DummyTraits = traits or DummyTraits()
    self.report: bool = report

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Docstring."""
    if name == "error":
      raise ValueError("bad type")
    return cst.Name(name)

  def _report_failure(self, msg: str) -> None:
    """Docstring."""
    pass

  def _get_target_traits(self) -> DummyTraits:
    """Docstring."""
    return self.traits


def test_handle_post_processing_branches() -> None:
  """Docstring."""
  # Base case
  rewriter: DummyRewriter = DummyRewriter()
  node: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_post_processing(rewriter, node, {}, "id") == node

  # 38 -> 39, 47 -> 59 (output_select_index success)
  node_tuple: cst.Subscript = getattr(getattr(cst.parse_statement("f()[0]"), "body")[0], "value")
  mapping_select: Dict[str, Any] = {"output_select_index": 0}
  handle_post_processing(rewriter, node_tuple, mapping_select, "id")
  # should be Subscript

  # 42 -> 43 -> 44 (output_select_index failure, with _report_failure)
  (cst.Pass())  # apply_index_select will fail if not expression maybe? Actually let's mock it or use bad index type
  mapping_bad_select: Dict[str, Any] = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, node, mapping_bad_select, "id")

  # 42 -> 44 (output_select_index failure, no _report_failure)
  class DummyRewriterNoReport(DummyRewriter):
    """Docstring."""

    def _report_failure(self, msg: str) -> None:
      """Docstring."""
      raise NotImplementedError()

  rewriter_no_report: DummyRewriterNoReport = DummyRewriterNoReport()
  del DummyRewriterNoReport._report_failure
  handle_post_processing(rewriter_no_report, node, mapping_bad_select, "id")

  # 47 -> 48 (output_cast) -> 59
  mapping_cast: Dict[str, Any] = {"output_cast": "float32"}
  handle_post_processing(rewriter, node, mapping_cast, "id")

  # 54 -> 55 (output_cast failure)
  mapping_bad_cast: Dict[str, Any] = {"output_cast": "error"}
  handle_post_processing(rewriter, node, mapping_bad_cast, "id")

  # 59 -> 65 (Signature stack is init module method)
  # 72 -> 73 (result is Call)
  # 74 -> 77 (auto strip true, has known magic)
  # 78 -> 79 (has keyword in magic -> force=True)
  # 86 -> 87 (inject)
  # 91 -> 92 (strip auto)
  # 96 -> 97 (strip)
  node_call: cst.Call = getattr(getattr(cst.parse_statement("f(key=1)"), "body")[0], "value")
  traits: DummyTraits = DummyTraits()
  traits.auto_strip_magic_args = True
  traits.strip_magic_args = ["strip_me"]
  traits.inject_magic_args = [("inj_me", "")]
  semantics: DummySemantics = DummySemantics(origins={"my_id": SemanticTier.NEURAL.value}, magic=["key"])
  context: DummyContext = DummyContext(is_init=True, is_module=True)
  rewriter_neural: DummyRewriter = DummyRewriter(context=context, semantics=semantics, traits=traits)

  handle_post_processing(rewriter_neural, node_call, {}, "my_id")

  # 72 -> 82 (Not a Call, but neural)
  node_not_call: cst.Name = cst.Name("x")
  handle_post_processing(rewriter_neural, node_not_call, {}, "my_id")

  # 78 -> 77 (keyword not in magic) -> 82 -> 99 (Not neural, force=False)
  node_call2: cst.Call = getattr(getattr(cst.parse_statement("f(other=1)"), "body")[0], "value")
  rewriter_not_neural: DummyRewriter = DummyRewriter(context=context, semantics=DummySemantics(origins={}), traits=traits)
  handle_post_processing(rewriter_not_neural, node_call2, {}, "other_id")

  # 84 -> 99 (is neural but result not call) - already tested above but let's be sure
  handle_post_processing(rewriter_neural, cst.Name("x"), {}, "my_id")

  # 74 -> 75 (auto_strip False)
  traits.auto_strip_magic_args = False
  handle_post_processing(rewriter_neural, node_call, {}, "my_id")
