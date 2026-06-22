from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_agent import install as install_mod
from fusion_agent.install import install_skill, print_config, skill_target_dir


def test_skill_target_dir_project(tmp_path: Path) -> None:
    result = skill_target_dir(scope="project", project_root=tmp_path)
    assert result == tmp_path / ".opencode" / "skills" / "fusion"


def test_skill_target_dir_global_opencode_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "custom"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert skill_target_dir(scope="global") == tmp_path / "custom" / "skills" / "fusion"


def test_skill_target_dir_global_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert skill_target_dir(scope="global") == tmp_path / "xdg" / "opencode" / "skills" / "fusion"


def test_skill_target_dir_unknown_scope() -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        skill_target_dir(scope="weird")


def test_install_skill_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod.Path, "cwd", staticmethod(lambda: tmp_path))
    target = install_skill(scope="project")
    assert target.exists()
    assert target.name == "SKILL.md"
    text = target.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: fusion" in text


def test_install_skill_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_mod.Path, "cwd", staticmethod(lambda: tmp_path))
    install_skill(scope="project")
    with pytest.raises(FileExistsError, match="already exists"):
        install_skill(scope="project")


def test_install_skill_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod.Path, "cwd", staticmethod(lambda: tmp_path))
    first = install_skill(scope="project")
    second = install_skill(scope="project", force=True)
    assert second == first
    assert second.exists()


def test_print_config_is_valid_json() -> None:
    data = json.loads(print_config())
    assert data["mcp"]["fusion"]["command"][0] == "uvx"
    assert data["mcp"]["fusion"]["enabled"] is True
    assert data["experimental"]["mcp_timeout"] == 90000


def test_bundled_skill_has_frontmatter() -> None:
    text = install_mod._bundled("skill.md")
    assert "name: fusion" in text
    assert "fusion_query" in text
