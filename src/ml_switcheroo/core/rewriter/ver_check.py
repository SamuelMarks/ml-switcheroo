"""
Versioning Check Mixin.

Handles parsing and verification of framework version strings against constraints
defined in the Semantic Knowledge Base.
"""

from typing import Optional, Tuple
import importlib.metadata
import re


class VersioningMixin:
  """
  Mixin for checking target framework version compatibility.

  Assumed attributes on self:
      target_fw (str): The target framework key.
      semantics (SemanticsManager): The knowledge base.
  """

  def __init__(self):
    self._cached_target_version: Optional[str] = None
    self._version_checked = False

  def _get_target_version(self) -> Optional[str]:
    """
    Resolves the version of the target framework.
    Prioritizes configuration overrides, then installed package metadata.
    """
    if self._version_checked:
      return self._cached_target_version

    self._version_checked = True

    # 1. Check if Semantic Config has a 'version' set (e.g. from Ghost Snapshot)
    fw_conf = self.semantics.get_framework_config(self.target_fw)
    if fw_conf and "version" in fw_conf:
      self._cached_target_version = fw_conf["version"]
      return self._cached_target_version

    # 2. Try importing live
    # Use mapping for weird package names (e.g. flax_nnx -> flax)
    pkg = self.target_fw
    if pkg == "flax_nnx":
      pkg = "flax"

    try:
      self._cached_target_version = importlib.metadata.version(pkg)
    except Exception:
      self._cached_target_version = None

    return self._cached_target_version

  def _parse_version(self, v_str: str) -> Tuple[int, ...]:
    """
    Parses version string into tuple of ints for comparison.
    Handles basic semver like '1.2.3' or '2.0.0+cuda'.
    """
    parts = []
    # Split by non-digits
    tokens = re.split(r"[^\d]+", v_str)
    for t in tokens:
      if t:
        parts.append(int(t))
    return tuple(parts)

  def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:
    """
    Verifies loaded target version against constraints.

    Args:
        min_v: Minimum required version string (inclusive).
        max_v: Maximum supported version string.

    Returns:
        None if compatible.
        String warning message if incompatible.
    """
    if not min_v and not max_v:
      return None

    current = self._get_target_version()
    if not current:
      # Can't verify if framework not installed/detected
      return None

    curr_tuple = self._parse_version(current)

    if min_v:
      min_tuple = self._parse_version(min_v)
      if curr_tuple < min_tuple:
        return f"Target {self.target_fw}@{current} is older than required {min_v}"

    if max_v:
      max_tuple = self._parse_version(max_v)
      # Treat max_version as loose upper bound (deprecated after this)
      if curr_tuple >= max_tuple:
        return f"Target {self.target_fw}@{current} exceeds max supported {max_v}"

    return None
