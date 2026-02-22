# SOP 07 — API & Output

## Goal
Expose the pipeline via FastAPI REST endpoints and WebSocket for live streaming.

## Endpoints

### 1. `POST /analyze/upload`

Upload a video or audio file for analysis.

**Request:**
```
Content-Type: multipart/form-data
Body: file (binary)
```

**Response (200):**
```json
{
  "session_id": "uuid",
  "violence_detected": true,
  "overall_alert": "Critical",
  "duration": 65.2,
  "total_chunks": 26,
  "events": [
    {
      "start": 14.2, "end": 17.1,
      "type": "abusive_speech",
      "confidence": 0.81,
      "alert": "Warning",
      "explanation": "Abusive language detected in speech"
    }
  ],
  "chunks": [
    {
      "chunk_id": 0, "start": 0.0, "end": 2.5,
      "fused_score": 0.12,
      "alert": "Safe",
      "transcript": ""
    }
  ],
  "escalation_trend": "rising",
  "temporal_prediction": "violence likely to escalate"
}
```

**Error Responses:**
- 400: Invalid file format
- 413: File too large
- 500: Processing error

### 2. `WebSocket /analyze/stream`

Real-time mic streaming analysis.

**Protocol:**
1. Client connects via WebSocket
2. Client sends binary audio frames (~0.5s, 16kHz mono, int16)
3. Server buffers until chunk_duration (2.5s)
4. Server processes chunk, sends JSON result:
```json
{
  "chunk_id": 5,
  "fused_score": 0.45,
  "alert": "Warning",
  "transcript": "I told you to stop",
  "event_type": "abusive_speech"
}
```
5. Client can close connection to stop

### 3. `GET /results/{session_id}`

Retrieve cached results from a previous upload analysis.

**Response:** Same JSON as `/analyze/upload`

## CORS Configuration
```python
origins = ["http://localhost:5173"]  # Vite dev server
```

## File Size Limits
- Max upload: 500MB
- Max audio duration: 30 minutes (configurable)

## Response Headers
- `Content-Type: application/json`
- Processing time in `X-Processing-Time` header
