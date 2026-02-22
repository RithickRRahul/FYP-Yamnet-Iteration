"""
Decision Engine — Alert level assignment and explanation generation.

Deterministic rules, no probabilistic inference.
Assigns: Safe 🟢 / Warning 🟡 / Critical 🔴
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def determine_chunk_alert(
    fused_score: float,
    temporal: dict,
    acoustic_events: list = None,
    has_speech: bool = False,
    nlp_threatening: bool = False,
) -> dict:
    """
    Determine alert level for a single chunk.

    Returns:
        {
            "alert": "Safe" | "Violence",
            "explanation": str,
            "reasons": list of str,
        }
    """
    reasons = []

    if fused_score > 0.3:
        reasons.append(f"Elevated violence score ({fused_score:.2f})")

    if temporal.get("trend") == "spike":
        reasons.append("Sudden spike in violence indicators")

    if temporal.get("trend") == "sustained" and fused_score > 0.7:
        reasons.append(f"Sustained aggression (score: {fused_score:.2f})")

    if temporal.get("trend") == "rising":
        reasons.append("Violence indicators are rising")

    if temporal.get("escalation_score", 0) > 0.3:
        reasons.append(f"Escalation detected (score: {temporal['escalation_score']:.2f})")

    if acoustic_events:
        event_names = [e.get("class", "unknown") for e in acoustic_events[:3]]
        reasons.append(f"Violent sounds detected: {', '.join(event_names)}")

    if nlp_threatening:
        reasons.append("Threatening language detected in speech")

    if reasons:
        return {
            "alert": "Violence",
            "explanation": "; ".join(reasons),
            "reasons": reasons,
        }

    # --- Safe ---
    return {
        "alert": "Safe",
        "explanation": "No violence indicators detected",
        "reasons": [],
    }


def determine_overall_alert(chunk_results: List[dict]) -> dict:
    """
    Determine overall alert level from all chunk results.

    Returns:
        {
            "violence_detected": bool,
            "overall_alert": "Safe" | "Violence",
            "total_chunks": int,
            "violence_chunks": int,
            "safe_chunks": int,
        }

    Improvements over teammate's version:
        - Chunk-level statistics
        - Based on sustained patterns, not single-chunk decisions
    """
    alerts = [r.get("alert", "Safe") for r in chunk_results]

    violence_count = alerts.count("Violence")
    safe_count = alerts.count("Safe")

    if violence_count > 0:
        overall = "Violence"
        violence_detected = True
    else:
        overall = "Safe"
        violence_detected = False

    return {
        "violence_detected": violence_detected,
        "overall_alert": overall,
        "total_chunks": len(alerts),
        "violence_chunks": violence_count,
        "safe_chunks": safe_count,
    }


def classify_event_type(
    acoustic_score: float,
    nlp_score: float,
    emotion_score: float,
    has_speech: bool,
    acoustic_events: list = None,
) -> str:
    """Determine the primary event type for this chunk."""
    if acoustic_events and acoustic_score > nlp_score:
        top_event = acoustic_events[0].get("class", "unknown") if acoustic_events else "unknown"
        event_lower = top_event.lower()
        if "gunshot" in event_lower or "gun" in event_lower:
            return "gunshot"
        elif "explosion" in event_lower or "boom" in event_lower:
            return "explosion"
        elif "scream" in event_lower or "shout" in event_lower:
            return "screaming"
        elif "glass" in event_lower or "shatter" in event_lower:
            return "glass_breaking"
        elif "slap" in event_lower or "smack" in event_lower:
            return "physical_violence"
        else:
            return "violent_sound"

    if has_speech and nlp_score > acoustic_score:
        return "abusive_speech"

    if has_speech and emotion_score > 0.6:
        return "aggressive_emotion"

    if acoustic_score > 0.3:
        return "violent_sound"

    return "combined"
