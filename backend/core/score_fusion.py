"""
Score Fusion — Combine acoustic, linguistic, and emotional signals.

Weighted score fusion with configurable weights.
Handles speech vs non-speech scenarios differently.
"""

import logging
import json
import os

logger = logging.getLogger(__name__)

# Default fusion weights
DEFAULT_WEIGHTS = {
    "w_acoustic": 0.5,
    "w_nlp": 0.3,
    "w_emotion": 0.2,
}


def load_optimized_weights(weights_path: str = None) -> dict:
    """Load optimized weights from file if available, else use defaults."""
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            weights = json.load(f)
        logger.info(f"Loaded optimized fusion weights: {weights}")
        return weights
    return DEFAULT_WEIGHTS.copy()


def fuse_scores(
    acoustic_score: float,
    nlp_score: float,
    emotion_score: float,
    has_speech: bool,
    weights: dict = None,
) -> dict:
    """
    Fuse multi-modal scores into a single violence probability.

    Args:
        acoustic_score: Acoustic violence score (0-1) from YAMNet
        nlp_score: NLP threat score (0-1) from Toxic BERT
        emotion_score: Emotion violence score (0-1) from emotion model
        has_speech: Whether speech was detected in this chunk
        weights: Custom weights dict (optional)

    Returns:
        {
            "fused_score": float (0-1),
            "component_scores": {
                "acoustic": float,
                "nlp": float,
                "emotion": float,
            },
            "fusion_mode": "full" | "acoustic_only",
            "weights_used": dict,
        }

    Improvements over teammate's version:
        - 3-signal fusion (teammate had only 2: acoustic + toxicity)
        - Handles no-speech mode (normalizes acoustic to full weight)
        - Configurable/optimizable weights
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    w_acoustic = weights.get("w_acoustic", 0.5)
    w_nlp = weights.get("w_nlp", 0.3)
    w_emotion = weights.get("w_emotion", 0.2)

    if has_speech:
        # Full 3-signal fusion
        fused = (
            w_acoustic * acoustic_score
            + w_nlp * nlp_score
            + w_emotion * emotion_score
        )
        fusion_mode = "full"
    else:
        # No speech — acoustic signal only, give it full weight
        # This prevents the score being artificially low when there's no speech
        fused = acoustic_score
        fusion_mode = "acoustic_only"

    # Clamp to [0, 1]
    fused = max(0.0, min(1.0, fused))

    return {
        "fused_score": round(fused, 4),
        "component_scores": {
            "acoustic": round(acoustic_score, 4),
            "nlp": round(nlp_score, 4) if has_speech else 0.0,
            "emotion": round(emotion_score, 4) if has_speech else 0.0,
        },
        "fusion_mode": fusion_mode,
        "weights_used": weights,
    }
