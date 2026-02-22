"""
Voice Activity Detection (VAD) — Silero VAD wrapper.

Determines if an audio chunk contains speech.
Returns speech probability and timestamp segments.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Lazy-loaded model (not at import time — avoids crashes if unavailable)
_model = None
_utils = None


def _load_model():
    """Lazy-load Silero VAD model on first use."""
    global _model, _utils
    if _model is None:
        logger.info("Loading Silero VAD model...")
        _model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        logger.info("Silero VAD loaded successfully.")
    return _model, _utils


def detect_speech(
    waveform: np.ndarray,
    sr: int = 16000,
    threshold: float = 0.5,
) -> dict:
    """
    Detect speech activity in an audio chunk.

    Args:
        waveform: 1D numpy array (16kHz mono, float32)
        sr: Sample rate
        threshold: Speech probability threshold (default 0.5)

    Returns:
        {
            "has_speech": bool,
            "speech_probability": float,
            "speech_timestamps": list of {start, end} in seconds,
        }

    Improvements over teammate's version:
        - Returns probability (not just boolean)
        - Returns timestamp segments
        - Lazy model loading (won't crash at import)
        - Error handling
    """
    try:
        model, utils = _load_model()
        get_speech_timestamps = utils[0]

        # Convert to torch tensor
        audio_tensor = torch.tensor(waveform, dtype=torch.float32)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor, model, sampling_rate=sr
        )

        # Calculate speech probability as fraction of audio that is speech
        total_samples = len(waveform)
        speech_samples = sum(
            ts["end"] - ts["start"] for ts in speech_timestamps
        )
        speech_probability = speech_samples / total_samples if total_samples > 0 else 0.0

        # Convert sample timestamps to seconds
        speech_segments = [
            {
                "start": round(ts["start"] / sr, 3),
                "end": round(ts["end"] / sr, 3),
            }
            for ts in speech_timestamps
        ]

        has_speech = speech_probability > threshold or len(speech_timestamps) > 0

        return {
            "has_speech": has_speech,
            "speech_probability": round(float(speech_probability), 4),
            "speech_timestamps": speech_segments,
        }

    except Exception as e:
        logger.error(f"VAD failed: {e}")
        # Fail safe: assume speech is present (better safe than sorry for violence detection)
        return {
            "has_speech": True,
            "speech_probability": 0.5,
            "speech_timestamps": [],
        }
