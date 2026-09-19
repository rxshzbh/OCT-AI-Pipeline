"""
serve/api.py — Stage 8: real-time serving.

Run with:  uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

Design choices that matter for latency:
- The OCTPipeline (all backbones, segmenters, heads) is built ONCE at
  module import time — @app.on_event("startup") — and reused for every
  request. Rebuilding it per-request would mean reloading RETFound/
  InternImage/nnU-Net/MedSAM weights every single call, which turns a
  few-second job into a multi-minute one.
- A single in-process asyncio.Lock serializes GPU access. A single GPU
  can't usefully run two forward passes at once anyway; the lock just
  stops two concurrent requests from racing on the same model state.
  If you need real concurrency, run multiple worker processes behind a
  load balancer instead of trying to parallelize inside one process.
- Report PDFs and JSON are written to disk with a UUID per request, so a
  doctor can be handed a stable link/ID rather than a raw response body.
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.infer_pipeline import OCTPipeline
from src.config import load_pipeline_config  # your existing config loader

UPLOAD_DIR = Path("data/incoming")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OCT AI Pipeline — Real-Time Inference")

_pipeline: OCTPipeline | None = None
_gpu_lock = asyncio.Lock()


@app.on_event("startup")
def load_models() -> None:
    """Runs exactly once when the server process starts, not per request."""
    global _pipeline
    cfg = load_pipeline_config()
    _pipeline = OCTPipeline(cfg)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly")

    request_id = str(uuid.uuid4())
    scan_path = UPLOAD_DIR / f"{request_id}_{file.filename}"
    scan_path.write_bytes(await file.read())

    async with _gpu_lock:
        # OCTPipeline.run() is synchronous CPU/GPU-bound work; run it in a
        # thread so we don't block the event loop for other requests
        # waiting on the lock (they'll still queue, but the server stays
        # responsive for health checks etc. in the meantime).
        result = await asyncio.to_thread(_pipeline.run, str(scan_path))

    if result["status"] != "ok":
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(
        content={
            "request_id": request_id,
            "status": "ok",
            "report_json": result["report_json"],
            "report_pdf_url": f"/report/{request_id}",
        }
    )


@app.get("/report/{request_id}")
async def get_report_pdf(request_id: str):
    matches = list(Path("data/reports").glob(f"{request_id}*.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(matches[0], media_type="application/pdf")


@app.get("/health")
async def health():
    return {"status": "ok" if _pipeline is not None else "loading"}
