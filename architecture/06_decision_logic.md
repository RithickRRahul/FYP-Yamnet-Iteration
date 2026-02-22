# SOP 06 — Decision Logic

## Goal
Assign a final alert level per chunk and overall, with human-readable explanations.

## Alert Level Definitions

| Level | Condition | Color |
|-------|-----------|-------|
| **Safe** 🟢 | No violence indicators detected | Green |
| **Violence** 🔴 | Elevated scores or identified violence trends | Red |

## Decision Rules (Deterministic)

```python
def determine_alert(fused_score, temporal):
    reasons = []
    
    if fused_score > 0.3:
        reasons.append(f"Elevated violence score ({fused_score:.2f})")
    if temporal["trend"] in ["spike", "rising"]:
        reasons.append("Rising violence indicators")
    
    if reasons:
        return "Violence"
    
    return "Safe"
```

## Overall Alert (Full File Analysis)

```python
# After all chunks are processed:
chunk_alerts = [chunk["alert"] for chunk in results]

if "Violence" in chunk_alerts:
    overall = "Violence"
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
    "alert": "Violence",
    "explanation": "Violent acoustic event detected: Gunshot"
}
```

## Output
```python
{
    "violence_detected": True,
    "overall_alert": "Violence",
    "events": [...],
    "escalation_trend": "rising",
    "temporal_prediction": "violence likely to escalate"
}
```
