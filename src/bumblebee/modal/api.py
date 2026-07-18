# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUntypedClassDecorator=false, reportUntypedFunctionDecorator=false
"""Bumblebee hosted API deployed on Modal (``modal deploy -m bumblebee.modal.api``).

One GPU container holds a started :class:`~bumblebee.engine.DocumentEngine` and
serves the FastAPI app from :mod:`bumblebee.api`. Deployment requires a Modal
secret named ``bumblebee-api`` holding either the compatible single-tenant
``BUMBLEBEE_API_KEY`` or the tenant mapping ``BUMBLEBEE_API_KEYS_JSON``::

    modal secret create bumblebee-api BUMBLEBEE_API_KEY=<token>
    bumblebee deploy-api          # or: modal deploy -m bumblebee.modal.api

Cost guardrails: the scaledown window is short (a warm GPU burns ~$2/hour) and
containers are capped at one. Cold starts take minutes (model load + vLLM
boot) — fine for demos; keep a request warm-up in mind before showing it live.
"""

import os

import modal

from bumblebee.modal.app import (
    CPU_CORES,
    MODAL,
    MODAL_SECRETS,
    app,
    hf_cache_vol,
    image,
    pilot_data_vol,
    vllm_cache_vol,
)

# Short idle window so demo deployments don't silently burn GPU credit.
API_SCALEDOWN_SECONDS = int(os.environ.get("BUMBLEBEE_API_SCALEDOWN_SECONDS", "120"))

api_image = image.uv_pip_install("fastapi[standard]>=0.115").env({"BUMBLEBEE_USAGE_DB": "/data/usage.sqlite3"})


@app.cls(
    image=api_image,
    gpu=MODAL.gpu,
    cpu=CPU_CORES,
    timeout=MODAL.timeout_seconds,
    startup_timeout=MODAL.startup_timeout_seconds,
    scaledown_window=API_SCALEDOWN_SECONDS,
    max_containers=1,
    secrets=[*MODAL_SECRETS, modal.Secret.from_name("bumblebee-api")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/data": pilot_data_vol,
    },
)
class ApiWorker:
    """One GPU container serving the bumblebee parse API."""

    @modal.enter()
    def start(self) -> None:
        """Load the layout model and start the vLLM OCR server."""
        from bumblebee.engine import DocumentEngine
        from bumblebee.logging import configure_logging

        configure_logging()
        self.engine = DocumentEngine().start()

    @modal.asgi_app()
    def api(self):  # noqa: D102 - Modal web-endpoint hook; docs live on build_api.
        from bumblebee.api import build_api

        return build_api(self.engine, usage_checkpoint=pilot_data_vol.commit)

    @modal.exit()
    def stop(self) -> None:
        """Tear down the vLLM server and worker threads."""
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.stop()
