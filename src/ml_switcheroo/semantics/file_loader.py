"""File Loading Logic for Semantic Knowledge Base.

This module handles the discovery and deserialization of JSON specification files
(The Hub) and Snapshot overlays (The Spokes). It delegates the actual data
combination logic to `ml_switcheroo.semantics.merging`.

Updates:
- Removed early exit if semantics directory is missing, enabling Snapshots to load independently.
- Using explicit path resolution imports.
"""

import json
import logging
import typing
from pathlib import Path

from pydantic import JsonValue

import ml_switcheroo

if typing.TYPE_CHECKING:
  import ml_switcheroo.semantics.manager

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.semantics.merging import infer_tier_from_priority, merge_overlay_data, merge_tier_data
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir

logger = logging.getLogger(__name__)

try:
  from ml_ecosystem_snapshots.utils import parse_docstring_with_griffe, resolve_griffe_parser, strip_sphinx_roles
except ImportError:
  import re

  def strip_sphinx_roles(text: str | None) -> str | None:
    """Strip Sphinx cross-referencing roles from text fallback.

    Args:
        text: The input text containing potential Sphinx roles.

    Returns:
        Cleaned text with Sphinx roles stripped, or None if input is None.
    """
    if text is None:
      return None
    return re.sub(r":[a-zA-Z0-9_:]+:`~?([^`]*)`", r"\1", str(text))

  def resolve_griffe_parser(parser_name: str) -> typing.Hashable | typing.Sized:
    """Resolve a parser identifier into a valid Griffe Parser instance fallback.

    Args:
        parser_name: The parser name ('rest', 'sphinx', 'google', 'numpy', 'auto').

    Returns:
        A Griffe Parser instance or None.
    """
    return None

  def parse_docstring_with_griffe(docstring: str, parser_name: str | None = None) -> list[typing.Hashable | typing.Sized]:
    """Parse a docstring using Griffe with dynamic fallback fallback.

    Args:
        docstring: The docstring text to parse.
        parser_name: Optional starting parser ('rest', 'google', 'numpy').

    Returns:
        A list of parsed Griffe DocstringSection objects.
    """
    return []


def clean_sphinx_roles(text: str | None) -> str | None:
  """Strip Sphinx cross-referencing roles from text using standard utility.

  Transforms roles such as `:class:`~torch.Tensor`` or `:func:`relu``
  into clean representations (e.g. `torch.Tensor` or `relu`).

  Args:
      text: The input text containing potential Sphinx roles.

  Returns:
      Cleaned text with Sphinx roles stripped, or None if input is None.
  """
  r = strip_sphinx_roles(text)
  return str(r) if r is not None else None


def parse_docstring_sections(docstring: str, parser_name: str | None = None) -> list[typing.Hashable | typing.Sized]:
  """Parse a docstring using unified Griffe parser extraction.

  Args:
      docstring: The docstring text to parse.
      parser_name: Optional starting parser ('rest', 'google', 'numpy').

  Returns:
      List of parsed Griffe DocstringSection objects.
  """
  res = parse_docstring_with_griffe(docstring, parser_name)
  assert isinstance(res, list)
  return res


def get_docstring_parser(parser_name: str) -> typing.Hashable | typing.Sized:
  """Resolve a parser identifier into a valid Griffe Parser instance.

  Args:
      parser_name: The parser name ('rest', 'sphinx', 'google', 'numpy', 'auto').

  Returns:
      A Griffe Parser instance or None.
  """
  res: typing.Hashable | typing.Sized = resolve_griffe_parser(parser_name)
  return res


def clean_content_descriptions(content: typing.Hashable | typing.Sized) -> None:
  """Clean Sphinx roles in operation descriptions across a content mapping.

  Args:
      content: Dictionary or collection of operation definitions.
  """
  if not isinstance(content, dict):
    return
  for _, op_data in content.items():
    if isinstance(op_data, dict):
      desc = op_data.get("description")
      if isinstance(desc, str):
        cleaned = clean_sphinx_roles(desc)
        op_data["description"] = cleaned or ""


# Filenames to treat as discovered/consensus content
DISCOVERED_FILENAMES = {"k_discovered.yaml"}


class KnowledgeBaseLoader:
  """Handle the I/O operations for populating the SemanticsManager."""

  def __init__(self, manager: "ml_switcheroo.semantics.manager.SemanticsManager") -> None:
    """Initialize the loader.

    Args:
        manager: The parent SemanticsManager instance to populate.

    """
    self.mgr = manager

  def load_knowledge_graph(self) -> None:
    """Scan the semantics directory for JSON specifications.

    Loads files in priority order:
    1. Array API (Math) - Priority 10
    2. Neural Net (Layers) - Priority 20
    3. Discovered/Extras - Priority 30
    """
    base_path = resolve_semantics_dir()

    # FIX: Do not return early if semantics dir missing.
    # Overlays (Snapshots) might still exist and need loading.
    if base_path.exists():
      import yaml

      all_files = list(base_path.rglob("*.yaml"))
      prioritized_files: list[tuple[int, Path]] = []

      for fpath in all_files:
        fname = fpath.name
        if fname == "schema.yaml":
          continue
        priority = 30
        if "array" in fname:
          priority = 10
        elif "neural" in fname:
          priority = 20
        elif fname in DISCOVERED_FILENAMES:
          priority = 20
        prioritized_files.append((priority, fpath))

      # Sort by priority
      prioritized_files.sort(key=lambda x: (x[0], x[1].name))

      for priority, fpath in prioritized_files:
        try:
          if fpath.suffix == ".yaml":
            with open(fpath, "r", encoding="utf-8") as f:
              yaml_content = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
              # Wrap it back into a dict keyed by operation name since atomic yamls are just the op definition
              if (
                isinstance(yaml_content, dict)
                and "operation" in yaml_content
                and isinstance(yaml_content["operation"], str)
                and len(yaml_content) < 10
                and not any(isinstance(v, dict) for k, v in yaml_content.items() if k not in ["variants", "std_args"])
              ):
                op_name = yaml_content["operation"]
                content = {op_name: yaml_content}
              else:
                content = yaml_content
          else:
            with open(fpath, "r", encoding="utf-8") as f:
              content = json.load(f)
          tier = infer_tier_from_priority(priority)
          self._load_tier_content(content, tier)
        except (OSError, json.JSONDecodeError, yaml.YAMLError, KeyError, ValueError, TypeError, AttributeError) as e:
          logger.warning("⚠️ Error loading %s: %s", fpath.name, e)

    # Load Overlays after specs (or even if specs missing)
    self._load_overlays()

  def _load_overlays(self) -> None:
    """Scan the snapshots directory for framework mapping overlays."""
    snap_dir = resolve_snapshots_dir()
    if not snap_dir.exists():
      return

    mapping_files = list(snap_dir.glob("*_map.json"))
    for fpath in mapping_files:
      try:
        with open(fpath, "r", encoding="utf-8") as f:
          content = json.load(f)
        self._load_overlay_content(content, fpath.name)
      except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("⚠️ Error loading overlay %s: %s", fpath.name, e)

  def _load_tier_content(self, content: typing.Hashable | typing.Sized, tier: SemanticTier) -> None:
    """Merge a specification dictionary into the manager.

    Args:
        content: The JSON content.
        tier: The Semantic Tier classification.

    """
    clean_content_descriptions(content)
    merge_tier_data(
      data=self.mgr.data,
      key_origins=self.mgr._key_origins,
      framework_configs=self.mgr.framework_configs,
      new_content=content,
      tier=tier,
    )

  def _load_overlay_content(
    self, content: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized], filename: str
  ) -> None:
    """Merge a snapshot overlay into the manager.

    Args:
        content: The JSON content.
        filename: The source filename (used for metadata inference).

    """
    clean_content_descriptions(content)
    merge_overlay_data(
      data=self.mgr.data,
      key_origins=self.mgr._key_origins,
      framework_configs=self.mgr.framework_configs,
      test_templates=self.mgr.test_templates,
      content=content,
      filename=filename,
    )
