# SOP 01 — Data Ingestion

## Goal
Accept video, audio, or live microphone input and produce a raw audio waveform for downstream processing.

## Supported Inputs

| Type | Formats | Handler |
|------|---------|---------|
| Video | `.mp4`, `.mov`, `.avi` | FFmpeg extracts audio track |
| Audio | `.wav`, `.mp3` | Direct load via `librosa` / `soundfile` |
| Live Mic | WebSocket stream | Buffered in real-time |

## Processing Steps

### File Upload
1. Validate file extension against allowed list
2. Save to `.tmp/{session_id}/input.*`
3. If video → extract audio via FFmpeg:
   ```bash
   ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output.wav
   ```
4. If audio → load directly
5. Return: `(waveform: np.ndarray, sample_rate: int)`

### Live Microphone
1. Frontend captures audio via Web Audio API (44100 Hz, mono)
2. Sends chunks over WebSocket (binary, ~0.5s buffers)
3. Backend accumulates into sliding window buffer
4. Processes when buffer reaches chunk_duration (2.5s)

## Error Handling
- Invalid format → HTTP 400 with message
- FFmpeg failure → HTTP 500, log error, clean up temp files
- Corrupt audio → catch librosa/soundfile exceptions, return error
- File too large → reject if > 500MB (configurable)

## Output
```python
{
    "waveform": np.ndarray,  # float32, shape (samples,)
    "sample_rate": 16000,
    "duration": float,       # seconds
    "source": "upload" | "mic"
}
```
