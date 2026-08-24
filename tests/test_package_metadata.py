"""Source-level packaging metadata consistency checks."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_apache_license_with_pep639():
    """Build metadata must agree with the repository's Apache license text."""
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert 'requires = ["hatchling>=1.27"]' in pyproject
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert "License :: OSI Approved" not in pyproject
    assert "Apache License" in license_text.splitlines()[0]
