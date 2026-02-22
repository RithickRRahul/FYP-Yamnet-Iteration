"""
NLP Toxicity Classifier — Toxic BERT wrapper.

Uses unitary/toxic-bert for multi-label toxicity classification.
Returns per-category scores (toxic, severe_toxic, threat, insult, etc.)
"""

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline
_pipeline = None

# Toxicity categories and their violence relevance weights
VIOLENCE_RELEVANT = {
    "toxic": 0.5,
    "severe_toxic": 0.9,
    "threat": 1.0,
    "insult": 0.4,
    "obscene": 0.3,
    "identity_hate": 0.6,
}


def _load_model():
    """Lazy-load Toxic BERT pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info("Loading Toxic BERT model...")
        _pipeline = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None,  # Return ALL category scores
            device=0,    # Use GPU
        )
        logger.info("Toxic BERT loaded successfully.")
    return _pipeline


def classify_toxicity(text: str) -> dict:
    """
    Classify text for toxicity and threat level.

    Args:
        text: Transcribed speech text

    Returns:
        {
            "nlp_threat_score": float (0-1), weighted violence-relevant score,
            "categories": {
                "toxic": float,
                "severe_toxic": float,
                "threat": float,
                "insult": float,
                "obscene": float,
                "identity_hate": float,
            },
            "is_threatening": bool,
            "raw_text": str,
        }

    Improvements over teammate's version:
        - Returns ALL toxicity categories (teammate returned only 1 score)
        - Weighted violence-relevant scoring (threat > insult > obscene)
        - Returns raw text for logging
        - Lazy loading
    """
    if not text or text.strip() == "":
        return {
            "nlp_threat_score": 0.0,
            "categories": {k: 0.0 for k in VIOLENCE_RELEVANT},
            "is_threatening": False,
            "raw_text": "",
        }

    try:
        pipe = _load_model()
        results = pipe(text[:512])  # Truncate to model max length

        # Parse results into category scores
        categories = {}
        if results and isinstance(results[0], list):
            for item in results[0]:
                label = item["label"].lower()
                categories[label] = round(float(item["score"]), 4)
        elif results:
            for item in results:
                label = item["label"].lower()
                categories[label] = round(float(item["score"]), 4)

        # Fill missing categories with 0
        for key in VIOLENCE_RELEVANT:
            if key not in categories:
                categories[key] = 0.0

        # Calculate violence-weighted NLP score
        # Prioritizes: threat > severe_toxic > toxic > identity_hate > insult > obscene
        weighted_score = 0.0
        total_weight = 0.0
        for cat, weight in VIOLENCE_RELEVANT.items():
            if cat in categories:
                weighted_score += categories[cat] * weight
                total_weight += weight

        nlp_threat_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Also check if any single high-violence category is significant
        threat_max = max(
            categories.get("threat", 0),
            categories.get("severe_toxic", 0),
        )
        nlp_threat_score = max(nlp_threat_score, threat_max)

        is_threatening = (
            categories.get("threat", 0) > 0.5
            or categories.get("severe_toxic", 0) > 0.5
            or nlp_threat_score > 0.6
        )

        return {
            "nlp_threat_score": round(min(nlp_threat_score, 1.0), 4),
            "categories": categories,
            "is_threatening": is_threatening,
            "raw_text": text,
        }

    except Exception as e:
        logger.error(f"Toxicity classification failed: {e}")
        return {
            "nlp_threat_score": 0.0,
            "categories": {k: 0.0 for k in VIOLENCE_RELEVANT},
            "is_threatening": False,
            "raw_text": text,
        }
