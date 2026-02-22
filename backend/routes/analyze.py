"""
Analyze Route — File upload and live mic streaming endpoints.

POST /analyze/upload — Upload video/audio for analysis
WebSocket /analyze/stream — Live microphone streaming
GET /results/{session_id} — Retrieve cached results
"""

import os
import uuid
import shutil
import logging
import numpy as np

from fastapi import APIRouter, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.core.pipeline import analyze_file, process_chunk
from backend.core.score_fusion import fuse_scores
from backend.core.temporal_analyzer import TemporalAnalyzer
from backend.core.decision_engine import determine_chunk_alert, classify_event_type
from backend.utils.audio_loader import (
    SUPPORTED_ALL,
    load_audio_from_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# In-memory results cache (use Redis in production)
_results_cache: dict = {}

# Temp directory for uploads
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tmp")
os.makedirs(TEMP_DIR, exist_ok=True)



@router.post("/upload")
async def analyze_upload(file: UploadFile):
    """
    Upload a video or audio file for violence detection analysis.

    Accepts: .mp4, .mov, .avi, .wav, .mp3, .flac, .ogg
    Returns: Full analysis with timestamps, events, and alert levels.

    Improvements over teammate's version:
        - Session ID for retrieving results later
        - Unique temp file names (no race conditions)
        - Proper file cleanup
        - Video support (FFmpeg extraction)
        - File format validation
        - Structured error responses
    """
    # Validate file format
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_ALL:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Supported: {', '.join(sorted(SUPPORTED_ALL))}",
        )

    # Save to temp with unique name
    session_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{session_id}{ext}")

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"[{session_id}] File uploaded: {file.filename} ({ext})")

        # Run full pipeline
        result = analyze_file(temp_path)
        result["session_id"] = session_id
        result["filename"] = file.filename

        # Cache result
        _results_cache[session_id] = result

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[{session_id}] Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"[{session_id}] Temp file cleaned up")


@router.websocket("/stream")
async def analyze_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time microphone streaming.

    Protocol:
        1. Client connects
        2. Client sends binary audio frames (16kHz mono, int16, ~0.5s buffers)
        3. Server buffers until chunk_duration (2.5s)
        4. Server processes and sends JSON result per chunk
        5. Client closes connection to stop

    This was COMPLETELY MISSING from teammate's code.
    """
    await websocket.accept()
    logger.info("WebSocket streaming session started")

    buffer = np.array([], dtype=np.float32)
    chunk_duration = 2.5
    sr = 16000
    chunk_samples = int(chunk_duration * sr)
    chunk_id = 0

    temporal_analyzer = TemporalAnalyzer()

    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()

            # Convert bytes to float32 array
            audio_chunk = load_audio_from_bytes(data, sr=sr)
            buffer = np.concatenate([buffer, audio_chunk])

            # Process when buffer reaches chunk size
            while len(buffer) >= chunk_samples:
                chunk_waveform = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # Process chunk
                result = process_chunk(chunk_waveform, sr=sr)

                # Temporal analysis
                temporal = temporal_analyzer.add_score(result["fused_score"])

                # Decision
                alert_info = determine_chunk_alert(
                    fused_score=result["fused_score"],
                    temporal=temporal,
                    acoustic_events=result.get("acoustic_events", []),
                    has_speech=result.get("has_speech", False),
                    nlp_threatening=result.get("is_threatening", False),
                )

                # Determine event type
                event_type = classify_event_type(
                    acoustic_score=result.get("acoustic_violence_score", 0),
                    nlp_score=result.get("nlp_threat_score", 0),
                    emotion_score=result.get("emotion_violence_score", 0),
                    has_speech=result.get("has_speech", False),
                    acoustic_events=result.get("acoustic_events", []),
                )

                # Send result to client
                response = {
                    "chunk_id": chunk_id,
                    "fused_score": result["fused_score"],
                    "acoustic_score": result.get("acoustic_violence_score", 0),
                    "nlp_score": result.get("nlp_threat_score", 0),
                    "emotion_score": result.get("emotion_violence_score", 0),
                    "has_speech": result.get("has_speech", False),
                    "transcript": result.get("transcript", ""),
                    "alert": alert_info["alert"],
                    "explanation": alert_info["explanation"],
                    "event_type": event_type,
                    "temporal": {
                        "trend": temporal["trend"],
                        "escalation_score": temporal["escalation_score"],
                        "prediction": temporal["prediction"],
                    },
                }

                await websocket.send_json(response)
                chunk_id += 1

    except WebSocketDisconnect:
        logger.info("WebSocket streaming session ended (client disconnected)")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=1011)


@router.get("/results/{session_id}")
async def get_results(session_id: str):
    """Retrieve cached analysis results by session ID."""
    if session_id not in _results_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for session: {session_id}",
        )
    return _results_cache[session_id]
