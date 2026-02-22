# SOP 02 — Preprocessing

## Goal
Convert raw audio to a normalized 16kHz mono waveform and split into overlapping chunks for analysis.

## Step 1: Format Standardization

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Channels | Mono (1) |
| Bit depth | float32 |
| Normalization | Peak normalize to -1.0 dB |

```python
# Resample if needed
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

# Mono conversion
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=0)

# Peak normalization
waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
```

## Step 2: Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk duration | 2.5 seconds | Enough for speech/events, small enough for real-time |
| Overlap | 0.5 seconds | Prevents cutting events at boundaries |
| Stride | 2.0 seconds | chunk_duration - overlap |
| Min final chunk | 1.0 seconds | Discard if shorter |

```python
chunk_samples = int(2.5 * 16000)  # 40,000 samples
stride_samples = int(2.0 * 16000) # 32,000 samples
```

## Output — Chunk Object
```python
{
    "chunk_id": int,
    "start_time": float,  # seconds
    "end_time": float,
    "waveform": np.ndarray,  # shape (40000,), float32
}
```

## Edge Cases
- Audio shorter than 2.5s → treat as single chunk, pad with zeros
- Silence-only audio → still process (VAD will flag as no speech)
- Very long audio (>30 min) → process in streaming fashion, limit memory
