# SPDX-License-Identifier: MIT

"""The version in pyproject.toml has release notes, indexed in docs/releases/README.md."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASES = ROOT / "docs" / "releases"


def _declared_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_the_installed_package_reports_the_declared_version() -> None:
    try:
        installed = version("agent-autosuggest-corpus")
    except PackageNotFoundError:  # not installed: pyproject.toml is the only source
        return
    assert installed == _declared_version()


def test_the_release_notes_exist_for_the_declared_version() -> None:
    current = _declared_version()
    notes = RELEASES / f"v{current}.md"
    assert notes.is_file(), f"{notes} is missing"
    heading = next(line for line in notes.read_text().splitlines() if line.startswith("# "))
    assert f"v{current}" in heading.split()
    assert f"(v{current}.md)" in (RELEASES / "README.md").read_text()
