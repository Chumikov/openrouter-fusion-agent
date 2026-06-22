"""Helper commands: ``install-skill`` and ``print-config``.

Reads bundled data files from the package (``fusion_agent/data/``) via
``importlib.resources`` and writes the skill into opencode's skill directory.
Path resolution is cross-OS and respects opencode's own overrides.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

SKILL_FILENAME = "SKILL.md"


def _bundled(name: str) -> str:
    """Read a file shipped inside the package under ``data/``."""
    return (files("fusion_agent") / "data" / name).read_text(encoding="utf-8")


def skill_target_dir(*, scope: str, project_root: Path | None = None) -> Path:
    """Resolve the opencode skill directory for ``scope``.

    * ``project``: ``<project_root>/.opencode/skills/fusion`` (opencode walks up
      the tree to find it, so this works on any OS).
    * ``global``: honours ``OPENCODE_CONFIG_DIR`` first, then ``XDG_CONFIG_HOME``,
      then falls back to ``~/.config/opencode``.
    """
    if scope == "project":
        root = project_root if project_root is not None else Path.cwd()
        return root / ".opencode" / "skills" / "fusion"
    if scope == "global":
        env_dir = os.environ.get("OPENCODE_CONFIG_DIR")
        if env_dir:
            return Path(env_dir) / "skills" / "fusion"
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            return Path(xdg) / "opencode" / "skills" / "fusion"
        return Path.home() / ".config" / "opencode" / "skills" / "fusion"
    raise ValueError(f"unknown scope: {scope!r}; expected 'project' or 'global'")


def install_skill(*, scope: str = "project", force: bool = False) -> Path:
    """Write ``SKILL.md`` into the resolved skill directory and return its path."""
    target_dir = skill_target_dir(scope=scope)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SKILL_FILENAME
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; re-run with --force to overwrite.")
    target.write_text(_bundled("skill.md"), encoding="utf-8")
    return target


def print_config() -> str:
    """Return the ready-to-paste ``opencode.json`` snippet."""
    return _bundled("opencode.json")
