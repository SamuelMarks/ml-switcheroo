"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
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


def test_post_branches() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()
  node: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")

  # 38 -> 47 (missing output_select_index)
  handle_post_processing(rewriter, node, {}, "id")

  # 47 -> 59 (missing output_cast)
  handle_post_processing(rewriter, node, {"output_select_index": None}, "id")

  # 38 -> 39, 47 -> 48 -> 59
  mapping: Dict[str, Any] = {"output_select_index": 0, "output_cast": "float32"}
  node_tuple: cst.Subscript = getattr(getattr(cst.parse_statement("f()[0]"), "body")[0], "value")
  handle_post_processing(rewriter, node_tuple, mapping, "id")

  # 59 -> 99 (no context signature stack)
  handle_post_processing(rewriter, node, mapping, "id")
