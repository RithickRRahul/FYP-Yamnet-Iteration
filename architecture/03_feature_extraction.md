# SOP 03 — Feature Extraction

## Goal
Extract multi-modal features from each audio chunk using 5 pretrained models.

---

## Model 1: Silero VAD (Voice Activity Detection)

**Purpose:** Determine if the chunk contains speech.

| Property | Value |
|----------|-------|
| Model | `silero_vad` (snakers4/silero-vad) |
| Input | 16kHz mono waveform |
| Output | `has_speech: bool`, `speech_probability: float`, `speech_timestamps: list` |
| Threshold | speech_probability > 0.5 → has_speech = True |

**Logic:**
- If `has_speech = True` → run Whisper, Toxic BERT, and Emotion model
- If `has_speech = False` → skip speech-dependent models, use only acoustic score

---

## Model 2: YAMNet (Acoustic Event Classification)

**Purpose:** Classify audio into 521 sound categories, map violent ones to an acoustic violence score.

| Property | Value |
|----------|-------|
| Model | YAMNet (TensorFlow Hub) |
| Input | 16kHz mono waveform |
| Output | class scores (521 classes), 1024-dim embeddings |

**Violent Class Mapping:**
```python
VIOLENT_CLASSES = [
    "Gunshot, gunfire", "Machine gun", "Explosion",
    "Screaming", "Shout", "Battle cry",
    "Glass (breaking)", "Shatter",
    "Slap, smack", "Whack, thwack",
    "Cap gun", "Fireworks", "Boom",
    "Crying, sobbing",  # context-dependent
]
```

**Scoring:**
```python
# Pretrained mode: max confidence across violent classes
acoustic_score = max(scores[violent_class_indices])

# Trained mode: LogReg on 1024-dim embeddings
acoustic_score = logreg_model.predict_proba(embeddings)[0][1]
```

---

## Model 3: Whisper (Speech-to-Text)

**Purpose:** Transcribe speech in the chunk.

| Property | Value |
|----------|-------|
| Model | `faster-whisper` (base model, ~500MB) |
| Input | 16kHz mono waveform |
| Output | `text: str`, `confidence: float` |
| Condition | Only runs if `has_speech = True` |

---

## Model 4: Toxic BERT (Text Toxicity)

**Purpose:** Score transcribed text for toxicity/threat level.

| Property | Value |
|----------|-------|
| Model | `unitary/toxic-bert` (HuggingFace) |
| Input | text string from Whisper |
| Output | `toxicity_score: float (0-1)` for categories: toxic, severe_toxic, obscene, threat, insult, identity_hate |
| Condition | Only runs if `has_speech = True` AND text is non-empty |

**Scoring:**
```python
# Use max of threat + severe_toxic + toxic scores
nlp_score = max(scores["threat"], scores["severe_toxic"], scores["toxic"])
```

---

## Model 5: Emotion Recognition

**Purpose:** Detect anger/fear from vocal tone.

| Property | Value |
|----------|-------|
| Model | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` or similar |
| Input | 16kHz mono waveform |
| Output | `emotions: {angry, fear, happy, sad, neutral, ...}` |
| Condition | Only runs if `has_speech = True` |

**Scoring:**
```python
emotion_score = max(emotions["angry"], emotions["fear"])
```

---

## Per-Chunk Feature Vector

```python
{
    "chunk_id": 0,
    "has_speech": True,
    "acoustic_violence_score": 0.72,
    "nlp_threat_score": 0.81,       # 0.0 if no speech
    "emotion_scores": {"angry": 0.6, "fear": 0.3},
    "emotion_violence_score": 0.6,  # 0.0 if no speech
    "transcript": "get out of here",
}
```
