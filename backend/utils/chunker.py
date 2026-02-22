"""
Audio Chunker — Split audio into overlapping time windows.

Per SOP 02: 2.5s chunks with 0.5s overlap (2.0s stride).
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class AudioChunk:
    """A single audio chunk for analysis."""
    chunk_id: int
    start_time: float
    end_time: float
    waveform: np.ndarray

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
        }


def chunk_audio(
    waveform: np.ndarray,
    sr: int = 16000,
    chunk_duration: float = 2.5,
    overlap: float = 0.5,
    min_chunk_duration: float = 1.0,
) -> List[AudioChunk]:
    """
    Split audio waveform into overlapping chunks.

    Args:
        waveform: 1D float32 numpy array (16kHz mono)
        sr: Sample rate (default 16000)
        chunk_duration: Duration of each chunk in seconds (default 2.5)
        overlap: Overlap between consecutive chunks in seconds (default 0.5)
        min_chunk_duration: Minimum duration for the last chunk (default 1.0)

    Returns:
        List of AudioChunk objects

    Note: Unlike teammate's version, this:
        - Adds overlap to prevent cutting events at boundaries
        - Zero-pads the last chunk instead of discarding it
        - Uses 2.5s chunks instead of 2.0s for better speech analysis
    """
    chunk_samples = int(chunk_duration * sr)
    stride_samples = int((chunk_duration - overlap) * sr)
    min_samples = int(min_chunk_duration * sr)
    total_samples = len(waveform)

    # Handle audio shorter than one chunk
    if total_samples <= chunk_samples:
        # Pad with zeros if needed
        if total_samples < chunk_samples:
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[:total_samples] = waveform
            waveform_padded = padded
        else:
            waveform_padded = waveform

        return [AudioChunk(
            chunk_id=0,
            start_time=0.0,
            end_time=total_samples / sr,
            waveform=waveform_padded,
        )]

    chunks = []
    chunk_id = 0
    start = 0

    while start < total_samples:
        end = start + chunk_samples

        if end <= total_samples:
            # Full chunk
            chunk_waveform = waveform[start:end]
        else:
            # Last chunk — check if it's long enough
            remaining = total_samples - start
            if remaining < min_samples:
                break  # Too short, skip

            # Zero-pad to full chunk size
            chunk_waveform = np.zeros(chunk_samples, dtype=np.float32)
            chunk_waveform[:remaining] = waveform[start:total_samples]

        chunks.append(AudioChunk(
            chunk_id=chunk_id,
            start_time=round(start / sr, 3),
            end_time=round(min(end, total_samples) / sr, 3),
            waveform=chunk_waveform,
        ))

        chunk_id += 1
        start += stride_samples

    return chunks
