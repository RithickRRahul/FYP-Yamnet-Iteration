# SOP 04 — Score Fusion

## Goal
Combine acoustic, linguistic, and emotional signals into a single violence probability per chunk.

## Weighted Score Fusion Formula

```python
violence_score = (
    w_acoustic * acoustic_violence_score
    + w_nlp * nlp_threat_score * has_speech
    + w_emotion * emotion_violence_score * has_speech
)
```

## Default Weights

| Weight | Value | Rationale |
|--------|-------|-----------|
| `w_acoustic` | 0.5 | Acoustic events (gunshots, screams) are strongest single signal |
| `w_nlp` | 0.3 | Verbal threats are strong but require speech |
| `w_emotion` | 0.2 | Emotion is supplementary — angry tone without context isn't violence |

## Speech vs Non-Speech Handling

| Scenario | Active Signals | Effective Formula |
|----------|---------------|-------------------|
| Speech present | All 3 | `0.5×acoustic + 0.3×nlp + 0.2×emotion` |
| No speech | Acoustic only | `0.5×acoustic` (max possible = 0.5 unless normalized) |

**Important:** When no speech is detected, normalize the acoustic score:
```python
if not has_speech:
    violence_score = acoustic_violence_score  # Full weight on acoustic
else:
    violence_score = w_a * acoustic + w_n * nlp + w_e * emotion
```

## Weight Optimization (Phase 3F)

After training, `optimize_weights.py` will grid search:
```python
w_acoustic ∈ [0.3, 0.4, 0.5, 0.6, 0.7]
w_nlp     ∈ [0.1, 0.2, 0.3, 0.4]
w_emotion ∈ [0.05, 0.1, 0.15, 0.2, 0.25]
# Constraint: w_acoustic + w_nlp + w_emotion = 1.0
```

Optimal weights saved to `models/fusion_weights.json`.

## Output
```python
{
    "chunk_id": 0,
    "fused_violence_score": 0.78,  # 0.0 - 1.0
    "component_scores": {
        "acoustic": 0.72,
        "nlp": 0.81,
        "emotion": 0.60
    }
}
```
