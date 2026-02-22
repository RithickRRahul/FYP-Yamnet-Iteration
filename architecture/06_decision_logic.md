# SOP 06 — Decision Logic

## Goal
Assign a final alert level per chunk and overall, with human-readable explanations.

## Alert Level Definitions

| Level | Condition | Color |
|-------|-----------|-------|
| **Safe** 🟢 | fused_score < 0.3 AND no escalation | Green |
| **Warning** 🟡 | fused_score 0.3–0.7 OR rising trend OR escalation_score > 0.3 | Yellow |
| **Critical** 🔴 | fused_score > 0.7 sustained 2+ chunks OR spike > 0.85 OR escalation_score > 0.7 | Red |

## Decision Rules (Deterministic)

```python
def determine_alert(fused_score, temporal):
    # Critical conditions
    if fused_score > 0.85:
        return "Critical"
    if temporal["trend"] == "spike":
        return "Critical"
    if temporal["trend"] == "sustained" and fused_score > 0.7:
        return "Critical"
    
    # Warning conditions
    if fused_score > 0.3:
        return "Warning"
    if temporal["trend"] == "rising":
        return "Warning"
    if temporal["escalation_score"] > 0.3:
        return "Warning"
    
    return "Safe"
```

## Overall Alert (Full File Analysis)

```python
# After all chunks are processed:
chunk_alerts = [chunk["alert"] for chunk in results]

if "Critical" in chunk_alerts:
    overall = "Critical"
elif "Warning" in chunk_alerts:
    overall = "Warning"
else:
    overall = "Safe"
```

## Explanation Generation

For each detected event, produce a human-readable explanation:

```python
EXPLANATIONS = {
    "acoustic_high": "Violent acoustic event detected: {class_name}",
    "nlp_high": "Abusive/threatening language detected in speech",
    "emotion_high": "High {emotion} detected in vocal tone",
    "spike": "Sudden spike in violence indicators",
    "escalation": "Violence indicators have been escalating over time",
    "sustained": "Sustained aggressive activity detected",
}
```

## Event Detection

An event is logged when `fused_score > 0.3`:

```python
{
    "start": chunk["start_time"],
    "end": chunk["end_time"],
    "type": "gunshot" | "abusive_speech" | "aggressive_emotion" | "combined",
    "confidence": fused_score,
    "alert": "Warning" | "Critical",
    "explanation": "Violent acoustic event detected: Gunshot"
}
```

## Output
```python
{
    "violence_detected": True,
    "overall_alert": "Critical",
    "events": [...],
    "escalation_trend": "rising",
    "temporal_prediction": "violence likely to escalate"
}
```
