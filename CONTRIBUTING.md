# Contributing to Violence Detection

Thank you for your interest in contributing to the Multimodal Violence Detection System! This document outlines how to get started.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/FYP_new_iteration_yamnet.git
   ```
3. Follow the Setup steps in `README.md` to ensure your local environment (Python + Node.js) works correctly.

## Contribution Guidelines

### Adding New Models

If you are replacing a model (e.g., swapping DistilBERT for a Llama-3 adapter):
- Place the wrapper in `backend/models/`.
- Maintain the exact return dictionaries expected by `backend/core/pipeline.py`.
- Update the scoring threshold variables if the new model outputs values on a different scale.

### Adjusting the Frontend

The frontend uses React, Vite, and TailwindCSS, customized with a `glass-panel` UI style in `index.css`.
- Core application state flows through `App.tsx`.
- Component edits belong in `frontend/src/components`.

### Pull Requests

1. Try to document the "why" heavily when opening Pull Requests.
2. If adjusting the core heuristic scoring in `temporal_analyzer.py`, please provide test logs proving it doesn't cause false positives on non-violent media.
3. Make sure to run `npm run build` in the frontend and resolve any TypeScript errors before committing.
