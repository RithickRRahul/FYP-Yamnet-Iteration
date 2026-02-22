"""
Emotion Detection — Speech emotion recognition.

Uses a pretrained wav2vec2 emotion model to detect anger, fear, and distress.
This was COMPLETELY MISSING from teammate's code.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Lazy-loaded model
_processor = None
_model = None
_device = None

# Model: superb/wav2vec2-base-superb-er (emotion recognition trained on IEMOCAP)
MODEL_NAME = "superb/wav2vec2-base-superb-er"

# Emotion labels from the model
EMOTION_LABELS = ["neutral", "happy", "angry", "sad"]

# Which emotions are violence-relevant
VIOLENCE_EMOTIONS = {"angry": 1.0, "sad": 0.2}


def _load_model():
    """Lazy-load emotion recognition model."""
    global _processor, _model, _device
    if _model is None:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

        logger.info(f"Loading emotion model: {MODEL_NAME}...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
        logger.info(f"Emotion model loaded on {_device}.")
    return _processor, _model, _device


def detect_emotion(
    waveform: np.ndarray,
    sr: int = 16000,
) -> dict:
    """
    Detect emotional state from audio waveform.

    Args:
        waveform: 1D numpy array (16kHz mono, float32)
        sr: Sample rate

    Returns:
        {
            "emotions": {"neutral": 0.1, "happy": 0.05, "angry": 0.7, "sad": 0.15},
            "dominant_emotion": "angry",
            "emotion_violence_score": float (0-1),
        }
    """
    try:
        processor, model, device = _load_model()

        # Process audio
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]

        # Map to emotion labels
        emotions = {}
        for i, label in enumerate(EMOTION_LABELS):
            if i < len(probs_np):
                emotions[label] = round(float(probs_np[i]), 4)

        # Dominant emotion
        dominant_idx = np.argmax(probs_np)
        dominant_emotion = EMOTION_LABELS[dominant_idx] if dominant_idx < len(EMOTION_LABELS) else "unknown"

        # Violence-relevant emotion score
        violence_score = 0.0
        for emotion, weight in VIOLENCE_EMOTIONS.items():
            if emotion in emotions:
                violence_score += emotions[emotion] * weight
        violence_score = min(violence_score, 1.0)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "emotion_violence_score": round(violence_score, 4),
        }

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")
        return {
            "emotions": {label: 0.0 for label in EMOTION_LABELS},
            "dominant_emotion": "unknown",
            "emotion_violence_score": 0.0,
        }
