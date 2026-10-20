"""FastAPI application factory.

Run in development::

    uvicorn verity_api.main:app --reload --port 8000

and start the Vite dev server separately (it proxies ``/api`` here). In
production, ``npm run build`` writes ``src/verity_ui/dist`` and this process
serves the SPA itself, so the whole tool is one port.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from verity_api.config import get_settings
from verity_api.routers import detectors, jobs, media, system
from verity_api.jobs import JobConflict
from verity_api.service import PipelineError

DESCRIPTION = """
HTTP surface for the Verity deepfake examination pipeline.

Every endpoint maps onto a `verity` CLI capability:

| CLI | API |
|---|---|
| `verity ingest` | `POST /api/media` |
| `verity extract` | `POST /api/extract` |
| `verity analyze` | `POST /api/analyze`, `GET /api/media/{id}` |
| `verity run` | `POST /api/run` |
| `verity compile` | `POST /api/compile` |
| `verity report` | `POST /api/report`, `GET /api/media/{id}/report` |
| `verity setup` | `POST /api/detectors/setup` |

Long-running operations return a job; poll `GET /api/jobs/{id}` or subscribe
to `GET /api/jobs/{id}/events`.
"""


def _dev_origins() -> list[str]:
    configured = os.environ.get("VERITY_CORS_ORIGINS", "")
    if configured:
        return [o.strip() for o in configured.split(",") if o.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Verity API",
        description=DESCRIPTION,
        version="0.4.1",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_dev_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(PipelineError)
    async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(JobConflict)
    async def _job_conflict(_: Request, exc: JobConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(system.router)
    app.include_router(detectors.router)
    app.include_router(media.router)
    app.include_router(jobs.router)

    @app.get("/api/health", tags=["system"])
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": app.version}

    _mount_ui(app)
    return app


def _mount_ui(app: FastAPI) -> None:
    """Serve the built SPA, with history-API fallback for client routes."""
    dist = get_settings().ui_dist
    if dist is None or not (dist / "index.html").exists():
        @app.get("/", include_in_schema=False)
        def _no_ui() -> Dict[str, str]:
            return {
                "detail": "UI is not built. Run `npm install && npm run build` in "
                "src/verity_ui, or use the Vite dev server on port 5173.",
                "api_docs": "/api/docs",
            }

        return

    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        # Real files (favicon, fonts, images) win; anything else is a route.
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app()


def main() -> None:
    """Console-script entry point: ``verity-server``."""
    import uvicorn

    uvicorn.run(
        "verity_api.main:app",
        host=os.environ.get("VERITY_HOST", "127.0.0.1"),
        port=int(os.environ.get("VERITY_PORT", "8000")),
        reload=os.environ.get("VERITY_RELOAD", "").lower() in ("1", "true", "yes"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
