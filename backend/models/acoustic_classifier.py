"""
Acoustic Violence Classifier — YAMNet-based sound event detection.

Mode 1 (Pretrained): Maps YAMNet's 521 classes to violent categories.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded models
_yamnet_model = None
_class_names = None

# Classes from YAMNet that indicate violent/aggressive sounds
VIOLENT_CLASSES = {
    "Gunshot, gunfire": 1.0,
    "Machine gun": 1.0,
    "Explosion": 0.95,
    "Screaming": 0.85,
    "Shout": 0.6,
    "Battle cry": 0.8,
    "Glass": 0.7,
    "Shatter": 0.8,
    "Breaking": 0.7,
    "Slap, smack": 0.9,
    "Whack, thwack": 0.85,
    "Cap gun": 0.7,
    "Boom": 0.75,
    "Crying, sobbing": 0.4,
    "Whimper": 0.35,
    "Siren": 0.3,
    "Fire alarm": 0.3,
    "Smash, crash": 0.75,
    "Thump, thud": 0.5,
}


def _load_yamnet():
    """Lazy-load YAMNet model."""
    global _yamnet_model, _class_names
    
    import tensorflow as tf
    import tensorflow_hub as hub
    
    if _yamnet_model is None:
        logger.info("Loading YAMNet model from TF Hub...")
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load class names
        class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
        import csv
        with open(class_map_path) as f:
            reader = csv.DictReader(f)
            _class_names = [row["display_name"] for row in reader]

        logger.info(f"YAMNet loaded. {len(_class_names)} sound classes available.")
    return _yamnet_model, _class_names



def extract_embeddings(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extract YAMNet embeddings and class scores from audio.

    Returns:
        (scores, embeddings, top_classes)
        - scores: (N, 521) class probabilities per frame
        - embeddings: (N, 1024) feature embeddings per frame
        - top_classes: list of (class_name, mean_score) sorted by score
    """
    model, class_names = _load_yamnet()
    import tensorflow as tf

    # YAMNet expects float32 waveform at 16kHz
    waveform_tf = tf.cast(waveform, tf.float32)
    scores, embeddings, spectrogram = model(waveform_tf)

    scores_np = scores.numpy()
    embeddings_np = embeddings.numpy()

    # For impulsive sounds (gunshots/punches), max score across frames is much more accurate than mean
    agg_scores = np.max(scores_np, axis=0)
    top_indices = np.argsort(agg_scores)[::-1][:10]
    top_classes = [
        (class_names[i], float(agg_scores[i]))
        for i in top_indices
    ]

    return agg_scores, embeddings_np, top_classes


def predict_acoustic_violence(
    waveform: np.ndarray,
) -> dict:
    """
    Predict acoustic violence score from audio.

    Uses mapped YAMNet classes.

    Returns:
        {
            "acoustic_violence_score": float (0-1),
            "detected_events": list of {"class": str, "score": float},
            "top_sounds": list of (class_name, score),
            "embeddings_mean": np.ndarray (1024-dim, for fusion),
            "mode": "pretrained",
        }
    """
    try:
        scores_np, embeddings_np, top_classes = extract_embeddings(waveform)
        model, class_names = _load_yamnet()

        # Mean embedding for downstream use
        embeddings_mean = np.mean(embeddings_np, axis=0)

        # Pretrained mode: map violent classes
        agg_scores = scores_np  # since we returned agg_scores tightly from extract_embeddings
        detected_events = []
        max_violence_score = 0.0

        for i, class_name in enumerate(class_names):
            # Check if this class matches any violent category
            for violent_name, violence_weight in VIOLENT_CLASSES.items():
                if violent_name.lower() in class_name.lower():
                    # Apply a 1.5x multiplier for isolated short-burst sounds to prevent dilution
                    boost = 1.5 if "gun" in violent_name.lower() or "slap" in violent_name.lower() or "boom" in violent_name.lower() else 1.0
                    weighted_score = float(agg_scores[i]) * violence_weight * boost
                    
                    if agg_scores[i] > 0.02:  # Lowered minimum detection threshold for faint sound effects
                        detected_events.append({
                            "class": class_name,
                            "score": round(float(agg_scores[i]), 4),
                            "violence_weight": violence_weight,
                        })
                    max_violence_score = max(max_violence_score, weighted_score)

        return {
            "acoustic_violence_score": round(min(max_violence_score, 1.0), 4),
            "detected_events": sorted(
                detected_events, key=lambda x: x["score"], reverse=True
            ),
            "top_sounds": top_classes[:5],
            "embeddings_mean": embeddings_mean,
            "mode": "pretrained",
        }

    except Exception as e:
        logger.error(f"Acoustic classification failed: {e}")
        return {
            "acoustic_violence_score": 0.0,
            "detected_events": [],
            "top_sounds": [],
            "embeddings_mean": np.zeros(1024),
            "mode": "error",
        }
