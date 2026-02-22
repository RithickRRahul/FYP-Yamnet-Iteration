"""
Temporal Analyzer — Rule-based sliding window trend detection.

Analyzes how violence scores evolve over consecutive chunks
to detect escalation, sustained aggression, and sudden spikes.
"""

import logging
from typing import List
from collections import deque

logger = logging.getLogger(__name__)

# Sliding window configuration
WINDOW_SIZE = 5  # ~12.5 seconds at 2.5s chunks


class TemporalAnalyzer:
    """
    Stateful temporal analysis over a stream of chunk scores.

    Improvements over teammate's version:
        - Sliding window (teammate had no window — just scanned all scores)
        - Multiple trend types (spike, rising, sustained, falling)
        - Escalation scoring (0-1, not just boolean)
        - Prediction capability ("violence likely to escalate")
        - Stateful for live streaming analysis
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.score_history: deque = deque(maxlen=window_size * 3)  # Keep more for context
        self.window: deque = deque(maxlen=window_size)

    def reset(self):
        """Reset state for a new analysis session."""
        self.score_history.clear()
        self.window.clear()

    def add_score(self, score: float) -> dict:
        """
        Add a new chunk score and analyze temporal trends.

        Args:
            score: Fused violence score for the latest chunk (0-1)

        Returns:
            {
                "trend": "stable" | "rising" | "falling" | "spike" | "sustained",
                "escalation_score": float (0-1),
                "prediction": str,
                "window_scores": list,
                "chunk_count": int,
            }
        """
        self.score_history.append(score)
        self.window.append(score)

        return self.analyze()

    def analyze(self) -> dict:
        """Analyze current window for temporal patterns."""
        scores = list(self.window)
        n = len(scores)

        if n < 2:
            return {
                "trend": "stable",
                "escalation_score": 0.0,
                "prediction": "insufficient data",
                "window_scores": scores,
                "chunk_count": n,
            }

        # --- Spike Detection ---
        is_spike = False
        if n >= 2:
            current = scores[-1]
            previous = scores[-2]
            is_spike = current > 0.85 and previous < 0.4

        # --- Rising Trend ---
        is_rising = False
        if n >= 3:
            recent = scores[-min(n, 5):]
            if len(recent) >= 3:
                increases = sum(
                    1 for i in range(len(recent) - 1)
                    if recent[i + 1] > recent[i]
                )
                delta = recent[-1] - recent[0]
                is_rising = increases >= len(recent) - 2 and delta > 0.15

        # --- Sustained Aggression ---
        is_sustained = sum(1 for s in scores if s > 0.5) >= 3

        # --- Falling Trend ---
        is_falling = False
        if n >= 3:
            recent = scores[-min(n, 5):]
            is_falling = recent[-1] < recent[0] - 0.2

        # --- Escalation Score ---
        escalation_score = 0.0
        if is_spike:
            escalation_score += 0.4
        if is_rising:
            escalation_score += 0.3
        if is_sustained:
            escalation_score += 0.3
        escalation_score = min(escalation_score, 1.0)

        # --- Determine Primary Trend ---
        if is_spike:
            trend = "spike"
        elif is_sustained:
            trend = "sustained"
        elif is_rising:
            trend = "rising"
        elif is_falling:
            trend = "falling"
        else:
            trend = "stable"

        # --- Prediction ---
        if trend == "spike":
            prediction = "sudden violent event detected"
        elif trend == "rising":
            prediction = "violence likely to escalate"
        elif trend == "sustained":
            prediction = "ongoing violent situation"
        elif trend == "falling":
            prediction = "situation de-escalating"
        else:
            prediction = "no escalation detected"

        return {
            "trend": trend,
            "escalation_score": round(escalation_score, 4),
            "prediction": prediction,
            "window_scores": [round(s, 4) for s in scores],
            "chunk_count": len(self.score_history),
        }


def detect_escalation(scores: List[float]) -> dict:
    """
    Stateless escalation detection for batch processing.
    Analyzes all scores at once (for file upload mode).
    """
    analyzer = TemporalAnalyzer()
    result = None
    for score in scores:
        result = analyzer.add_score(score)
    return result if result else {
        "trend": "stable",
        "escalation_score": 0.0,
        "prediction": "no data",
        "window_scores": [],
        "chunk_count": 0,
    }
