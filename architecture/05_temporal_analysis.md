# SOP 05 — Temporal Analysis

## Goal
Analyze how violence scores change over consecutive chunks to detect escalation patterns and predict future violence.

## Method: Rule-Based Sliding Window

No trained LSTM — pure deterministic logic on the recent score history.

### Sliding Window Parameters

| Parameter | Value |
|-----------|-------|
| Window size | 5 chunks (~12.5 seconds at 2.5s chunks) |
| Update frequency | Every new chunk |

## Trend Detection Rules

### 1. Spike Detection
```python
is_spike = current_score > 0.85 and previous_score < 0.4
```
**Meaning:** Sudden jump from safe to dangerous. Immediate attention needed.

### 2. Rising Trend
```python
recent_scores = window[-5:]
is_rising = all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1))
           and (recent_scores[-1] - recent_scores[0]) > 0.15
```
**Meaning:** Consistent escalation over time. Predictive signal for "violence likely soon."

### 3. Sustained Aggression
```python
is_sustained = sum(1 for s in window if s > 0.5) >= 3
```
**Meaning:** 3+ chunks above threshold = ongoing violent situation.

### 4. Falling Trend
```python
is_falling = recent_scores[-1] < recent_scores[0] - 0.2
```
**Meaning:** Situation is de-escalating.

## Escalation Score
```python
escalation_score = 0.0
if is_spike: escalation_score += 0.4
if is_rising: escalation_score += 0.3
if is_sustained: escalation_score += 0.3
# Clamp to [0, 1]
```

## Output
```python
{
    "trend": "rising" | "falling" | "sustained" | "spike" | "stable",
    "escalation_score": 0.7,
    "prediction": "violence likely to escalate" | "situation de-escalating" | "stable",
    "window_scores": [0.2, 0.35, 0.5, 0.62, 0.78]
}
```
