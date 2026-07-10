"""CLI smoke tests: --help must work without the GPU stack installed."""

import re

from typer.testing import CliRunner

from bumblebee.cli import app

runner = CliRunner()
HELP_ENV = {"COLUMNS": "120"}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain(text: str) -> str:
    """Remove Rich ANSI styling from help output."""
    return ANSI_RE.sub("", text)


def test_help_works_without_gpu_stack():
    result = runner.invoke(app, ["--help"], env=HELP_ENV)
    assert result.exit_code == 0
    output = plain(result.output)
    assert "--source" in output
    assert "--target" in output


def test_modal_help():
    result = runner.invoke(app, ["modal", "--help"], env=HELP_ENV)
    assert result.exit_code == 0
    assert "--detach" in plain(result.output)
