"""
Backend Test Script -- Verify all models load and pipeline works end-to-end.

Generates a synthetic test audio and runs it through the full pipeline.
"""

import sys
import os
import time
import numpy as np

# Set recursion limit for TensorFlow
sys.setrecursionlimit(10000)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_audio(duration=5.0, sr=16000):
    """Generate a synthetic audio clip with speech-like characteristics."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Mix of frequencies to simulate complex audio
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)    # Low tone
        + 0.2 * np.sin(2 * np.pi * 800 * t)   # Mid tone
        + 0.1 * np.sin(2 * np.pi * 3000 * t)  # High tone
        + 0.05 * np.random.randn(len(t))       # Noise
    ).astype(np.float32)

    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio, sr


def test_chunker():
    print("\n" + "=" * 60)
    print("TEST 1: Audio Chunker")
    print("=" * 60)

    from backend.utils.chunker import chunk_audio

    audio, sr = generate_test_audio(duration=7.0)
    chunks = chunk_audio(audio, sr=sr)

    print(f"  Audio duration: {len(audio) / sr:.1f}s")
    print(f"  Chunks created: {len(chunks)}")
    for c in chunks:
        print(f"    Chunk {c.chunk_id}: {c.start_time:.2f}s - {c.end_time:.2f}s "
              f"(samples: {len(c.waveform)})")

    assert len(chunks) >= 3, f"Expected >= 3 chunks, got {len(chunks)}"
    assert chunks[0].waveform.shape[0] == int(2.5 * sr), "Chunk size mismatch"
    print("  [PASS] Chunker OK")


def test_vad():
    print("\n" + "=" * 60)
    print("TEST 2: Silero VAD")
    print("=" * 60)

    from backend.models.vad import detect_speech

    audio, sr = generate_test_audio(duration=2.5)
    start = time.time()
    result = detect_speech(audio, sr=sr)
    elapsed = time.time() - start

    print(f"  Has speech: {result['has_speech']}")
    print(f"  Speech probability: {result['speech_probability']}")
    print(f"  Segments: {result['speech_timestamps']}")
    print(f"  Time: {elapsed:.2f}s")
    assert "has_speech" in result
    print("  [PASS] VAD OK")


def test_yamnet():
    print("\n" + "=" * 60)
    print("TEST 3: YAMNet Acoustic Classifier")
    print("=" * 60)

    from backend.models.acoustic_classifier import predict_acoustic_violence

    audio, sr = generate_test_audio(duration=2.5)
    start = time.time()
    result = predict_acoustic_violence(audio)
    elapsed = time.time() - start

    print(f"  Acoustic violence score: {result['acoustic_violence_score']}")
    print(f"  Mode: {result['mode']}")
    print(f"  Detected events: {result['detected_events'][:3]}")
    print(f"  Top sounds: {result['top_sounds'][:3]}")
    print(f"  Embedding shape: {result['embeddings_mean'].shape}")
    print(f"  Time: {elapsed:.2f}s")
    assert 0 <= result["acoustic_violence_score"] <= 1
    assert result["embeddings_mean"].shape == (1024,)
    print("  [PASS] YAMNet OK")


def test_whisper():
    print("\n" + "=" * 60)
    print("TEST 4: Faster-Whisper Transcriber")
    print("=" * 60)

    from backend.models.transcriber import transcribe

    audio, sr = generate_test_audio(duration=2.5)
    start = time.time()
    result = transcribe(audio, sr=sr)
    elapsed = time.time() - start

    print(f"  Transcript: '{result['text']}'")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Language: {result['language']}")
    print(f"  Time: {elapsed:.2f}s")
    assert isinstance(result["text"], str)
    print("  [PASS] Whisper OK")


def test_toxicity():
    print("\n" + "=" * 60)
    print("TEST 5: Toxic BERT NLP Classifier")
    print("=" * 60)

    from backend.models.nlp_classifier import classify_toxicity

    start = time.time()
    result_safe = classify_toxicity("Hello, how are you today?")
    elapsed1 = time.time() - start

    print(f"  Safe text score: {result_safe['nlp_threat_score']}")
    print(f"  Categories: {result_safe['categories']}")

    start = time.time()
    result_threat = classify_toxicity("I will hurt you, get out now!")
    elapsed2 = time.time() - start

    print(f"  Threat text score: {result_threat['nlp_threat_score']}")
    print(f"  Is threatening: {result_threat['is_threatening']}")
    print(f"  Time: {elapsed1:.2f}s / {elapsed2:.2f}s")

    assert result_threat["nlp_threat_score"] >= result_safe["nlp_threat_score"], \
        "Threatening text should score higher"
    print("  [PASS] Toxicity OK")


def test_emotion():
    print("\n" + "=" * 60)
    print("TEST 6: Emotion Detector")
    print("=" * 60)

    from backend.models.emotion_detector import detect_emotion

    audio, sr = generate_test_audio(duration=2.5)
    start = time.time()
    result = detect_emotion(audio, sr=sr)
    elapsed = time.time() - start

    print(f"  Emotions: {result['emotions']}")
    print(f"  Dominant: {result['dominant_emotion']}")
    print(f"  Violence score: {result['emotion_violence_score']}")
    print(f"  Time: {elapsed:.2f}s")
    assert "emotions" in result
    print("  [PASS] Emotion OK")


def test_fusion():
    print("\n" + "=" * 60)
    print("TEST 7: Score Fusion")
    print("=" * 60)

    from backend.core.score_fusion import fuse_scores

    result1 = fuse_scores(0.7, 0.8, 0.6, has_speech=True)
    print(f"  Speech mode: {result1['fused_score']} ({result1['fusion_mode']})")

    result2 = fuse_scores(0.7, 0.0, 0.0, has_speech=False)
    print(f"  No-speech mode: {result2['fused_score']} ({result2['fusion_mode']})")

    assert result1["fused_score"] > 0, "Fused score should be positive"
    assert result2["fusion_mode"] == "acoustic_only"
    print("  [PASS] Fusion OK")


def test_temporal():
    print("\n" + "=" * 60)
    print("TEST 8: Temporal Analyzer")
    print("=" * 60)

    from backend.core.temporal_analyzer import TemporalAnalyzer

    analyzer = TemporalAnalyzer()

    scores = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
    for score in scores:
        result = analyzer.add_score(score)

    print(f"  Trend: {result['trend']}")
    print(f"  Escalation: {result['escalation_score']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Window: {result['window_scores']}")

    assert result["trend"] in ("rising", "sustained", "spike")
    print("  [PASS] Temporal OK")


def test_decision():
    print("\n" + "=" * 60)
    print("TEST 9: Decision Engine")
    print("=" * 60)

    from backend.core.decision_engine import determine_chunk_alert, determine_overall_alert

    alert1 = determine_chunk_alert(0.9, {"trend": "stable", "escalation_score": 0.0})
    print(f"  Score 0.9: {alert1['alert']} -- {alert1['explanation']}")

    alert2 = determine_chunk_alert(0.5, {"trend": "rising", "escalation_score": 0.3})
    print(f"  Score 0.5 + rising: {alert2['alert']} -- {alert2['explanation']}")

    alert3 = determine_chunk_alert(0.1, {"trend": "stable", "escalation_score": 0.0})
    print(f"  Score 0.1: {alert3['alert']} -- {alert3['explanation']}")

    assert alert1["alert"] == "Critical"
    assert alert2["alert"] == "Warning"
    assert alert3["alert"] == "Safe"

    overall = determine_overall_alert([
        {"alert": "Safe"}, {"alert": "Warning"}, {"alert": "Critical"}, {"alert": "Safe"}
    ])
    print(f"  Overall: {overall['overall_alert']} (violence: {overall['violence_detected']})")
    assert overall["overall_alert"] == "Critical"
    print("  [PASS] Decision OK")


if __name__ == "__main__":
    print("BACKEND PIPELINE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Chunker", test_chunker),
        ("VAD", test_vad),
        ("YAMNet", test_yamnet),
        ("Whisper", test_whisper),
        ("Toxicity", test_toxicity),
        ("Emotion", test_emotion),
        ("Fusion", test_fusion),
        ("Temporal", test_temporal),
        ("Decision", test_decision),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  [FAIL]: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  FAIL {name}: {err}")
    else:
        print("\nALL TESTS PASSED!")
    print("=" * 60)
