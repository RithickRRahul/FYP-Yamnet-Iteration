"""
FastAPI Application — Main entry point.

Endpoints:
    POST /analyze/upload — File upload analysis
    WebSocket /analyze/stream — Live mic streaming
    GET /results/{session_id} — Retrieve cached results
"""

import os
import sys
import logging
import time

# Allow standard python application initialization without forcing recursion depth which crashes C-runtime on Windows.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes.analyze import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🛡️ AI Violence Detection System",
    description=(
        "Multimodal real-time violence and abusive speech detection. "
        "Analyzes audio/video using YAMNet (acoustic), Whisper + Toxic BERT (speech), "
        "and emotion recognition with weighted score fusion."
    ),
    version="2.0.0",
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Next.js / React dev
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "app": "AI Violence Detection System",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /analyze/upload",
            "stream": "WebSocket /analyze/stream",
            "results": "GET /results/{session_id}",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}
