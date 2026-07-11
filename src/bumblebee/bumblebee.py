"""Bumblebee: the RAG-ingestion product CLI on the bumblebee engine.

Same commands and flags as ``bumblebee`` (run locally, or ``bumblebee modal``),
with chunk emission defaulted **on**: every document gets a RAG-ready
``chunks.jsonl`` beside its markdown. Explicit ``--no-chunks`` flags and
pre-set ``BUMBLEBEE_EMIT_CHUNKS`` environment values still win.
"""

import os
import shutil
import subprocess

import typer

from bumblebee.cli import app


@app.command(name="deploy-api")
def deploy_api() -> None:
    """Deploy the bumblebee hosted API to Modal (persistent web endpoint).

    Requires the ``modal`` CLI and a Modal secret named ``bumblebee-api``
    holding ``BUMBLEBEE_API_KEY``. Engine/Modal settings come from the ambient
    ``BUMBLEBEE_*`` environment variables, like every other run mode.
    """
    if shutil.which("modal") is None:
        raise typer.BadParameter(
            "The `modal` CLI is not installed. Install the Modal extra: pip install 'bumblebee[modal]'."
        )
    result = subprocess.run(["modal", "deploy", "-m", "bumblebee.modal.api"], check=False)
    raise typer.Exit(code=result.returncode)


def main() -> None:
    """Run the bumblebee CLI (the ``bumblebee`` console script)."""
    # The env default is how the shared config resolves unset flags, so setting
    # it here flips the product default without forking the CLI.
    os.environ.setdefault("BUMBLEBEE_EMIT_CHUNKS", "1")
    app()


if __name__ == "__main__":
    main()
