"""
Pipeline Orchestrator — Chains all tools in the correct order.

Input → Load → Chunk → [VAD → Acoustic → Whisper → NLP → Emotion] → Fusion → Temporal → Decision → Output
"""

import os
import uuid
import time
import logging
from typing import Optional

from backend.utils.audio_loader import load_audio
from backend.utils.chunker import chunk_audio
from backend.models.vad import detect_speech
from backend.models.acoustic_classifier import predict_acoustic_violence
from backend.models.transcriber import transcribe
from backend.models.nlp_classifier import classify_toxicity
from backend.models.emotion_detector import detect_emotion
from backend.core.score_fusion import fuse_scores
from backend.core.temporal_analyzer import TemporalAnalyzer
from backend.core.decision_engine import (
    determine_chunk_alert,
    determine_overall_alert,
    classify_event_type,
)

logger = logging.getLogger(__name__)


def analyze_file(
    file_path: str,
    fusion_weights: Optional[dict] = None,
) -> dict:
    """
    Full pipeline analysis of an audio/video file.

    Args:
        file_path: Path to audio (.wav/.mp3) or video (.mp4/.mov/.avi) file
        fusion_weights: Optional custom fusion weights

    Returns:
        Full analysis result matching gemini.md schema
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{session_id}] Starting analysis of: {file_path}")

    # Step 1: Load audio
    waveform, sr = load_audio(file_path)
    duration = len(waveform) / sr
    logger.info(f"[{session_id}] Audio loaded: {duration:.1f}s")

    # Step 2: Chunk
    chunks = chunk_audio(waveform, sr=sr)
    logger.info(f"[{session_id}] Split into {len(chunks)} chunks")

    # Step 3: Process each chunk
    chunk_results = []
    events = []
    temporal_analyzer = TemporalAnalyzer()

    for chunk in chunks:
        chunk_result = process_chunk(
            chunk.waveform,
            sr=sr,
            fusion_weights=fusion_weights,
        )

        # Add chunk timing info
        chunk_result["chunk_id"] = chunk.chunk_id
        chunk_result["start_time"] = chunk.start_time
        chunk_result["end_time"] = chunk.end_time

        # Temporal analysis
        temporal = temporal_analyzer.add_score(chunk_result["fused_score"])
        chunk_result["temporal"] = temporal

        # Decision for this chunk
        alert_info = determine_chunk_alert(
            fused_score=chunk_result["fused_score"],
            temporal=temporal,
            acoustic_events=chunk_result.get("acoustic_events", []),
            has_speech=chunk_result.get("has_speech", False),
            nlp_threatening=chunk_result.get("is_threatening", False),
        )
        chunk_result["alert"] = alert_info["alert"]
        chunk_result["explanation"] = alert_info["explanation"]

        # Log events (when alert is Warning or Critical)
        if alert_info["alert"] != "Safe":
            event_type = classify_event_type(
                acoustic_score=chunk_result.get("acoustic_violence_score", 0),
                nlp_score=chunk_result.get("nlp_threat_score", 0),
                emotion_score=chunk_result.get("emotion_violence_score", 0),
                has_speech=chunk_result.get("has_speech", False),
                acoustic_events=chunk_result.get("acoustic_events", []),
            )
            events.append({
                "start": chunk.start_time,
                "end": chunk.end_time,
                "type": event_type,
                "confidence": chunk_result["fused_score"],
                "alert": alert_info["alert"],
                "explanation": alert_info["explanation"],
                "transcript": chunk_result.get("transcript", ""),
            })

        chunk_results.append(chunk_result)

    # Step 4: Overall assessment
    overall = determine_overall_alert(
        [{"alert": r["alert"]} for r in chunk_results]
    )

    # Final temporal analysis
    final_temporal = temporal_analyzer.analyze()

    processing_time = time.time() - start_time
    logger.info(
        f"[{session_id}] Analysis complete: "
        f"{overall['overall_alert']}, {len(events)} events, "
        f"{processing_time:.2f}s"
    )

    return {
        "session_id": session_id,
        "violence_detected": overall["violence_detected"],
        "overall_alert": overall["overall_alert"],
        "duration": round(duration, 2),
        "total_chunks": len(chunks),
        "events": events,
        "chunks": [
            {
                "chunk_id": r["chunk_id"],
                "start": r["start_time"],
                "end": r["end_time"],
                "fused_score": r["fused_score"],
                "acoustic_score": r.get("acoustic_violence_score", 0),
                "nlp_score": r.get("nlp_threat_score", 0),
                "emotion_score": r.get("emotion_violence_score", 0),
                "has_speech": r.get("has_speech", False),
                "transcript": r.get("transcript", ""),
                "alert": r["alert"],
                "explanation": r["explanation"],
            }
            for r in chunk_results
        ],
        "temporal_analysis": {
            "escalation_trend": final_temporal["trend"],
            "escalation_score": final_temporal["escalation_score"],
            "prediction": final_temporal["prediction"],
        },
        "statistics": {
            "violence_chunks": overall["violence_chunks"],
            "safe_chunks": overall["safe_chunks"],
        },
        "processing_time": round(processing_time, 2),
    }


def process_chunk(
    waveform,
    sr: int = 16000,
    fusion_weights: dict = None,
) -> dict:
    """
    Process a single audio chunk through all models.

    Returns per-chunk analysis results.
    """
    # 1. VAD — check for speech
    vad_result = detect_speech(waveform, sr=sr)
    has_speech = vad_result["has_speech"]

    # 2. Acoustic — always run (works on speech AND non-speech)
    acoustic_result = predict_acoustic_violence(
        waveform
    )
    acoustic_score = acoustic_result["acoustic_violence_score"]

    # 3. Speech-dependent models (only if speech detected)
    transcript = ""
    nlp_score = 0.0
    is_threatening = False
    nlp_categories = {}
    emotion_score = 0.0
    emotions = {}

    if has_speech:
        # Whisper transcription
        transcription = transcribe(waveform, sr=sr)
        transcript = transcription["text"]

        # NLP toxicity (only if transcription has text)
        if transcript.strip():
            nlp_result = classify_toxicity(transcript)
            nlp_score = nlp_result["nlp_threat_score"]
            is_threatening = nlp_result["is_threatening"]
            nlp_categories = nlp_result["categories"]

        # Emotion detection
        emotion_result = detect_emotion(waveform, sr=sr)
        emotion_score = emotion_result["emotion_violence_score"]
        emotions = emotion_result["emotions"]

    # 4. Score fusion
    fusion = fuse_scores(
        acoustic_score=acoustic_score,
        nlp_score=nlp_score,
        emotion_score=emotion_score,
        has_speech=has_speech,
        weights=fusion_weights,
    )

    return {
        "has_speech": has_speech,
        "speech_probability": vad_result["speech_probability"],
        "acoustic_violence_score": acoustic_score,
        "acoustic_events": acoustic_result["detected_events"],
        "acoustic_mode": acoustic_result["mode"],
        "transcript": transcript,
        "nlp_threat_score": nlp_score,
        "is_threatening": is_threatening,
        "nlp_categories": nlp_categories,
        "emotion_violence_score": emotion_score,
        "emotions": emotions,
        "fused_score": fusion["fused_score"],
        "fusion_mode": fusion["fusion_mode"],
    }
