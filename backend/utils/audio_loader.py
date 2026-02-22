"""
Audio Loader — Load audio from files or extract from video.

Handles: .wav, .mp3, .mp4, .mov, .avi
Outputs: 16kHz mono float32 numpy array
"""

import os
import subprocess
import tempfile
import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_ALL = SUPPORTED_AUDIO | SUPPORTED_VIDEO
TARGET_SR = 16000


def validate_file(file_path: str) -> str:
    """Validate file exists and has supported extension. Returns extension."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_ALL:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_ALL))}"
        )

    file_size = os.path.getsize(file_path)
    max_size = 500 * 1024 * 1024  # 500MB
    if file_size > max_size:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: 500MB"
        )

    return ext


def extract_audio_from_video(video_path: str, output_dir: str = None) -> str:
    """
    Extract audio from video using FFmpeg.
    Returns path to extracted .wav file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="violence_det_")

    output_path = os.path.join(output_dir, "extracted_audio.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(TARGET_SR),    # 16kHz
        "-ac", "1",               # Mono
        "-y",                     # Overwrite
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {result.returncode}): {result.stderr[:500]}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and add to PATH: "
            "https://ffmpeg.org/download.html"
        )

    logger.info(f"Extracted audio from video: {output_path}")
    return output_path


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load any supported audio/video file and return normalized 16kHz mono waveform.

    Returns:
        (waveform: np.ndarray float32, sample_rate: int)
    """
    ext = validate_file(file_path)

    # If video, extract audio first
    if ext in SUPPORTED_VIDEO:
        logger.info(f"Video detected ({ext}), extracting audio...")
        audio_path = extract_audio_from_video(file_path)
    else:
        audio_path = file_path

    # Load audio with librosa (auto-resamples to target SR)
    try:
        waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    # Ensure float32
    waveform = waveform.astype(np.float32)

    # Peak normalization (prevent division by zero)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    duration = len(waveform) / sr
    logger.info(
        f"Loaded audio: {duration:.1f}s, {sr}Hz, "
        f"samples={len(waveform)}, peak_normalized"
    )

    return waveform, sr


def load_audio_from_bytes(audio_bytes: bytes, sr: int = TARGET_SR) -> np.ndarray:
    """
    Load audio from raw bytes (for WebSocket mic streaming).
    Expects 16-bit PCM mono at the given sample rate.
    """
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_array = audio_array / 32768.0  # Normalize int16 to [-1, 1]

    # Resample if needed
    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

    return audio_array
