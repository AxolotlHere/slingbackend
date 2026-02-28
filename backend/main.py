"""
backend/main.py

PULSE Backend — FastAPI Application

Entrypoint for the PULSE backend server.

Endpoints:
    WebSocket  /ws/{session_id}     — Real-time sensor data ingestion
    GET        /health              — Health check
    GET        /report/{session_id} — Generate session PDF report
    GET        /session/{session_id}/summary — Session aggregate stats

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Active pipeline sessions: session_id → PULSEPipeline
_active_pipelines: dict = {}
_active_managers: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("PULSE Backend starting up...")
    # Create output directory
    Path("output/reports").mkdir(parents=True, exist_ok=True)
    yield
    logger.info("PULSE Backend shutting down. Active sessions: "
                + str(list(_active_pipelines.keys())))


app = FastAPI(
    title="PULSE — Physical Understanding of Living Street Economics",
    description="Road condition assessment backend: 5-channel sensors + 6 AI agents.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow PWA collector from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    import torch
    cuda_available = False
    cuda_device = None
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status":         "ok",
        "version":        "1.0.0",
        "cuda_available": cuda_available,
        "cuda_device":    cuda_device,
        "active_sessions": list(_active_pipelines.keys()),
    }


# ── WebSocket — Real-Time Data Ingestion ───────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for smartphone sensor data.

    The PWA collector connects here and streams:
        IMU packets  @ 200Hz
        GPS packets  @ 1Hz
        FRAME packets @ 2fps (640×360 base64 JPEG)
        AUDIO packets @ 10Hz (RMS + optional raw samples)

    Every complete 100m segment is processed through the full agent pipeline
    and the result is streamed back as SEGMENT_COMPLETE.
    """
    await websocket.accept()
    logger.info(f"WebSocket session opened: {session_id}")

    # Initialise pipeline and segment manager for this session
    from backend.pipeline import PULSEPipeline
    from backend.segment_manager import SegmentManager

    pipeline = PULSEPipeline(session_id)
    manager  = SegmentManager(session_id)

    _active_pipelines[session_id] = pipeline
    _active_managers[session_id]  = manager

    # Acknowledge session start
    await websocket.send_json({
        "type":       "SESSION_STARTED",
        "session_id": session_id,
        "message":    "PULSE backend ready. Stream sensor data.",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                packet = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Session {session_id}: invalid JSON packet received")
                continue

            # Ingest into segment buffer
            await manager.ingest(packet)

            # Process any completed segments
            while manager.segment_ready():
                segment = manager.pop_segment()
                try:
                    result = await pipeline.process_segment(segment)
                    await websocket.send_json({
                        "type":    "SEGMENT_COMPLETE",
                        "segment": _serialise_result(result),
                    })
                except Exception as exc:
                    logger.error(f"Segment processing error: {exc}", exc_info=True)
                    await websocket.send_json({
                        "type":       "SEGMENT_ERROR",
                        "segment_id": segment.get("segment_id"),
                        "error":      str(exc),
                    })

    except WebSocketDisconnect:
        logger.info(f"WebSocket session closed: {session_id}")

    except Exception as exc:
        logger.error(f"WebSocket session {session_id} error: {exc}", exc_info=True)

    finally:
        # Finalise any remaining buffered data
        last_seg = manager.finalise_session()
        if last_seg:
            try:
                result = await pipeline.process_segment(last_seg)
                _active_pipelines.pop(session_id, None)
                _active_managers.pop(session_id, None)
                pipeline.finalise()
                logger.info(f"Session {session_id} finalised. "
                            f"Segments: {manager.total_segments}")
            except Exception:
                pass


# ── REST Endpoints ─────────────────────────────────────────────────────────

@app.get("/session/{session_id}/summary", tags=["Sessions"])
async def get_session_summary(session_id: str):
    """Return aggregate statistics for a completed or active session."""
    pipeline = _active_pipelines.get(session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return pipeline.get_session_summary()


@app.get("/report/{session_id}", tags=["Reports"])
async def generate_report(session_id: str):
    """
    Generate and return a PDF report for a completed session.
    """
    pipeline = _active_pipelines.get(session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    summary = pipeline.get_session_summary()
    if summary.get("segments_processed", 0) == 0:
        raise HTTPException(status_code=400, detail="No segments processed in this session yet.")

    try:
        from backend.output.report_generator import generate_pdf_report
        pdf_path = generate_pdf_report(summary, output_dir="output/reports")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"PULSE_report_{session_id}.pdf",
        )
    except Exception as exc:
        logger.error(f"Report generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all currently active sessions."""
    return {
        "active_sessions": [
            {
                "session_id": sid,
                "segments":   p.get_session_summary().get("segments_processed", 0),
            }
            for sid, p in _active_pipelines.items()
        ]
    }


# ── Utilities ──────────────────────────────────────────────────────────────

def _serialise_result(result: dict) -> dict:
    """
    Convert result dict to JSON-serializable form.
    Removes numpy arrays and Open3D objects — keep only scalars/lists/strings.
    """
    import numpy as np

    clean = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            clean[k] = v.item()
        elif k in ("point_cloud",):
            # Open3D point cloud — skip for WebSocket (too large)
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = _serialise_result(v)
        elif isinstance(v, list):
            clean[k] = [
                _serialise_result(i) if isinstance(i, dict) else i
                for i in v
            ]
        else:
            clean[k] = v
    return clean
