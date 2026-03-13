import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict

import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compress import CompressionEngine

UPLOAD_DIR = ROOT / "uploads"
COMPRESSED_DIR = ROOT / "compressed"
CONFIG_PATH = ROOT / "config" / "results.json"
STATIC_DIR = ROOT / "src" / "api" / "static"
UPLOAD_PATH = UPLOAD_DIR / "original_model.pt"

MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = FastAPI(title="Model Compression API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _get_engine() -> CompressionEngine:
    return CompressionEngine(
        upload_path=UPLOAD_PATH,
        compressed_dir=COMPRESSED_DIR,
        results_path=CONFIG_PATH,
    )


def _serialize_event(payload: Dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/upload")
async def upload_model(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pt", ".pth"}:
        raise HTTPException(status_code=400, detail="Only .pt or .pth files are allowed.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 500MB limit.")

    with UPLOAD_PATH.open("wb") as f:
        f.write(content)

    try:
        engine = _get_engine()
        engine.load_original_model()
        info = engine.model_info()
    except Exception as exc:
        UPLOAD_PATH.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid PyTorch model file: {exc}") from exc

    return {
        "filename": file.filename,
        "saved_path": str(UPLOAD_PATH),
        "model_info": {
            "size_mb": info["size_mb"],
            "parameter_count": info["parameter_count"],
            "layer_count": info["layer_count"],
            "input_shape": info["input_shape"],
        },
    }


async def _compression_stream() -> AsyncGenerator[str, None]:
    if not UPLOAD_PATH.exists():
        yield _serialize_event({"status": "error", "message": "No uploaded model found."})
        return

    try:
        yield _serialize_event({"step": "quantization", "status": "running", "progress": 5})
        engine = _get_engine()
        await asyncio.to_thread(engine.load_original_model)

        await asyncio.to_thread(engine.run_dynamic_quantization)
        yield _serialize_event({"step": "quantization", "status": "done", "progress": 25})

        yield _serialize_event({"step": "pruning", "status": "running", "progress": 50})
        await asyncio.to_thread(engine.run_pruning)
        yield _serialize_event({"step": "pruning", "status": "done", "progress": 50})

        yield _serialize_event({"step": "distillation", "status": "running", "progress": 75})
        await asyncio.to_thread(engine.run_distillation)
        yield _serialize_event({"step": "distillation", "status": "done", "progress": 75})

        yield _serialize_event({"step": "onnx", "status": "running", "progress": 90})
        results = await asyncio.to_thread(engine.select_and_save_best)
        yield _serialize_event({"step": "onnx", "status": "done", "progress": 100})
        yield _serialize_event({"status": "done", "results": results})
    except Exception as exc:
        yield _serialize_event({"status": "error", "message": str(exc)})


@app.post("/api/compress")
async def compress_model_post():
    return StreamingResponse(_compression_stream(), media_type="text/event-stream")


@app.get("/api/compress")
async def compress_model_get():
    return StreamingResponse(_compression_stream(), media_type="text/event-stream")


@app.get("/api/results")
async def get_results():
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="No results available yet.")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@app.get("/api/download/compressed")
async def download_compressed_pt():
    path = COMPRESSED_DIR / "compressed_model.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Compressed .pt model not found.")
    return FileResponse(path, filename="compressed_model.pt", media_type="application/octet-stream")


@app.get("/api/download/onnx")
async def download_compressed_onnx():
    path = COMPRESSED_DIR / "compressed_model.onnx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Compressed .onnx model not found.")
    return FileResponse(path, filename="compressed_model.onnx", media_type="application/octet-stream")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "upload_exists": UPLOAD_PATH.exists(),
        "compressed_exists": (COMPRESSED_DIR / "compressed_model.pt").exists(),
    }
